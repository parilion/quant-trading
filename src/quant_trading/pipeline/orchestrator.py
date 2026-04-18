from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os
import time
from uuid import uuid4
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from quant_trading.config import Settings
from quant_trading.db.engine import make_engine
from quant_trading.db.init_db import init_db
from quant_trading.pipeline.backtest import topk_backtest
from quant_trading.pipeline.features import build_features, build_label_and_split
from quant_trading.pipeline.model import fit_and_predict

STAGES = [
    "universe_snapshot",
    "raw_ingest",
    "clean_align",
    "label_build",
    "dataset_split",
    "fit_predict",
    "backtest",
    "evaluate_report",
]


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}Z] {msg}", flush=True)


def _format_progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _format_eta(elapsed_sec: float, done: int, total: int) -> str:
    if done <= 0 or total <= done:
        return "00:00:00" if total <= done else "--:--:--"
    remain = elapsed_sec / done * (total - done)
    h = int(remain // 3600)
    m = int((remain % 3600) // 60)
    s = int(remain % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _make_run_id() -> str:
    return f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _to_tushare_date(date_str: str) -> str:
    return date_str.replace("-", "")


def _load_tushare_client(token: str, base_url: str):
    if os.environ.get("TUSHARE_IGNORE_SYSTEM_PROXY", "1") == "1":
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
            os.environ.pop(key, None)

    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("tushare is required for --execute mode. Install dependency: pip install tushare") from exc
    ts.set_token(token)
    pro = ts.pro_api()
    # Proxy address required by http://tsy.xiaodefa.cn/docs.html
    pro._DataApi__http_url = base_url
    pro._DataApi__timeout = int(os.environ.get("TUSHARE_HTTP_TIMEOUT_SEC", "60"))
    return pro


class _RateLimiter:
    def __init__(self, max_calls: int, period_sec: float) -> None:
        self.max_calls = max_calls
        self.period_sec = period_sec
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and (now - self._timestamps[0]) >= self.period_sec:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                wait_sec = self.period_sec - (now - self._timestamps[0]) + 0.01
            if wait_sec > 0:
                time.sleep(wait_sec)


_TS_RATE_LIMITER = _RateLimiter(
    max_calls=int(os.environ.get("TUSHARE_MAX_CALLS_PER_MIN", "120")),
    period_sec=60.0,
)
_TS_CLIENT_LOCAL = threading.local()


def _call_with_retry(callable_fn, retries: int = 6, delay_sec: float = 2.0):
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _TS_RATE_LIMITER.acquire()
            return callable_fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == retries:
                raise
            time.sleep(delay_sec * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected retry execution state.")


def _get_thread_tushare_client(settings: Settings):
    pro = getattr(_TS_CLIENT_LOCAL, "pro_client", None)
    if pro is None:
        pro = _load_tushare_client(settings.tushare_token, settings.tushare_base_url)
        _TS_CLIENT_LOCAL.pro_client = pro
    return pro


def _fetch_raw_day(
    trade_date: str,
    target_universe: set[str],
    settings: Settings,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    pro = _get_thread_tushare_client(settings)
    daily_day = _call_with_retry(lambda d=trade_date: pro.daily(trade_date=d))
    adj_day = _call_with_retry(lambda d=trade_date: pro.adj_factor(trade_date=d))
    basic_day = _call_with_retry(
        lambda d=trade_date: pro.daily_basic(
            trade_date=d,
            fields="ts_code,trade_date,pe_ttm,pb,ps_ttm,dv_ttm",
        )
    )

    out_daily = pd.DataFrame()
    out_basic = pd.DataFrame()

    if daily_day is not None and not daily_day.empty:
        daily_day = daily_day[daily_day["ts_code"].isin(target_universe)].copy()
        if not daily_day.empty:
            if adj_day is not None and not adj_day.empty:
                adj_day = adj_day[adj_day["ts_code"].isin(target_universe)].copy()
                daily_day = daily_day.merge(
                    adj_day[["ts_code", "trade_date", "adj_factor"]],
                    on=["ts_code", "trade_date"],
                    how="left",
                )
            else:
                daily_day["adj_factor"] = pd.NA
            daily_day["trade_date"] = pd.to_datetime(daily_day["trade_date"]).dt.date
            daily_day["is_suspended"] = 0
            out_daily = daily_day[
                ["trade_date", "ts_code", "open", "high", "low", "close", "vol", "amount", "adj_factor", "is_suspended"]
            ].drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

    if basic_day is not None and not basic_day.empty:
        basic_day = basic_day[basic_day["ts_code"].isin(target_universe)].copy()
        if not basic_day.empty:
            basic_day["trade_date"] = pd.to_datetime(basic_day["trade_date"]).dt.date
            out_basic = basic_day[["trade_date", "ts_code", "pe_ttm", "pb", "ps_ttm", "dv_ttm"]].drop_duplicates(
                subset=["trade_date", "ts_code"],
                keep="last",
            )

    return trade_date, out_daily, out_basic


def _log_stage(
    engine: Engine,
    run_id: str,
    stage: str,
    status: str,
    start_time: datetime,
    end_time: datetime | None = None,
    error_msg: str | None = None,
) -> None:
    sql = text(
        """
        INSERT INTO meta_run_log (run_id, stage, status, start_time, end_time, error_msg)
        VALUES (:run_id, :stage, :status, :start_time, :end_time, :error_msg)
        ON DUPLICATE KEY UPDATE
          status = VALUES(status),
          end_time = VALUES(end_time),
          error_msg = VALUES(error_msg),
          update_time = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "run_id": run_id,
                "stage": stage,
                "status": status,
                "start_time": start_time,
                "end_time": end_time,
                "error_msg": error_msg,
            },
        )


def _upsert_dataframe(
    engine: Engine,
    table_name: str,
    frame: pd.DataFrame,
    key_cols: list[str],
    cleanup_where: str = "",
    cleanup_params: dict[str, object] | None = None,
) -> int:
    if frame.empty:
        return 0
    with engine.begin() as conn:
        if cleanup_where:
            conn.execute(text(f"DELETE FROM {table_name} WHERE {cleanup_where}"), cleanup_params or {})  # noqa: S608
        else:
            keys = frame[key_cols].drop_duplicates()
            clauses: list[str] = []
            params: dict[str, object] = {}
            for idx, row in keys.iterrows():
                sub = []
                for col in key_cols:
                    pname = f"{col}_{idx}"
                    params[pname] = row[col]
                    sub.append(f"{col} = :{pname}")
                clauses.append("(" + " AND ".join(sub) + ")")
            if clauses:
                conn.execute(text(f"DELETE FROM {table_name} WHERE " + " OR ".join(clauses)), params)  # noqa: S608
    frame.to_sql(table_name, engine, if_exists="append", index=False)
    return int(len(frame))


def _stage_universe_snapshot(settings: Settings, engine: Engine) -> dict[str, object]:
    pro = _load_tushare_client(settings.tushare_token, settings.tushare_base_url)
    start_date = _to_tushare_date(settings.run_start_date)
    end_date = _to_tushare_date(settings.run_end_date)
    weights = _call_with_retry(
        lambda: pro.index_weight(index_code=settings.universe_index, start_date=start_date, end_date=end_date)
    )
    if weights is None or weights.empty:
        raise RuntimeError("No index_weight data fetched from Tushare for configured date range.")
    frame = (
        weights[["trade_date", "index_code", "con_code"]]
        .drop_duplicates()
        .rename(columns={"con_code": "ts_code"})
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["in_out_flag"] = 1
    rows = _upsert_dataframe(engine, "meta_universe", frame, ["trade_date", "index_code", "ts_code"])
    return {"status": "ok", "rows": rows}


def _stage_raw_ingest(settings: Settings, engine: Engine) -> dict[str, object]:
    pro = _get_thread_tushare_client(settings)
    with engine.begin() as conn:
        universe_df = pd.read_sql(
            text(
                """
                SELECT DISTINCT ts_code
                FROM meta_universe
                WHERE index_code = :index_code
                ORDER BY ts_code
                """
            ),
            conn,
            params={"index_code": settings.universe_index},
        )
    if universe_df.empty:
        raise RuntimeError("meta_universe is empty. Run universe_snapshot first.")

    start_date = _to_tushare_date(settings.run_start_date)
    end_date = _to_tushare_date(settings.run_end_date)
    trade_cal = _call_with_retry(lambda: pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date))
    if trade_cal is None or trade_cal.empty:
        raise RuntimeError("No trading calendar fetched from Tushare.")
    open_days = trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"].astype(str).tolist()
    if not open_days:
        raise RuntimeError("No open trading days in configured range.")

    target_universe = set(universe_df["ts_code"].tolist())
    batch_days = int(os.environ.get("RAW_INGEST_BATCH_DAYS", "15"))
    workers = int(os.environ.get("RAW_INGEST_WORKERS", "8"))
    day_chunk = int(os.environ.get("RAW_INGEST_DAY_CHUNK", "180"))
    processed_days = 0
    total_daily_rows = 0
    total_basic_rows = 0
    buffer_daily: list[pd.DataFrame] = []
    buffer_basic: list[pd.DataFrame] = []
    ingest_started = time.monotonic()

    def flush_buffers() -> tuple[int, int]:
        if not buffer_daily and not buffer_basic:
            return 0, 0
        daily_rows = 0
        basic_rows = 0
        if buffer_daily:
            daily_df = pd.concat(buffer_daily, ignore_index=True)
            daily_rows = _upsert_dataframe(engine, "ods_daily_bar", daily_df, ["trade_date", "ts_code"])
            buffer_daily.clear()
        if buffer_basic:
            basic_df = pd.concat(buffer_basic, ignore_index=True)
            basic_rows = _upsert_dataframe(engine, "ods_fundamental", basic_df, ["trade_date", "ts_code"])
            buffer_basic.clear()
        return daily_rows, basic_rows

    for chunk_start in range(0, len(open_days), day_chunk):
        day_slice = open_days[chunk_start : chunk_start + day_chunk]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fetch_raw_day, trade_date, target_universe, settings)
                for trade_date in day_slice
            ]
            for future in as_completed(futures):
                _, daily_day, basic_day = future.result()
                if not daily_day.empty:
                    buffer_daily.append(daily_day)
                if not basic_day.empty:
                    buffer_basic.append(basic_day)
                processed_days += 1
                if processed_days % batch_days == 0:
                    d_rows, b_rows = flush_buffers()
                    total_daily_rows += d_rows
                    total_basic_rows += b_rows
                    elapsed = max(time.monotonic() - ingest_started, 1.0)
                    est_rpm = (processed_days * 3) / elapsed * 60.0
                    progress_bar = _format_progress_bar(processed_days, len(open_days))
                    percent = processed_days / len(open_days) * 100.0
                    eta = _format_eta(elapsed, processed_days, len(open_days))
                    _log(
                        f"raw_ingest {progress_bar} {percent:6.2f}% ({processed_days}/{len(open_days)} days), "
                        f"daily_rows={total_daily_rows}, fundamental_rows={total_basic_rows}, "
                        f"est_req_per_min={est_rpm:.1f}, eta={eta}"
                    )

    d_rows, b_rows = flush_buffers()
    total_daily_rows += d_rows
    total_basic_rows += b_rows
    _log(
        f"raw_ingest {_format_progress_bar(processed_days, len(open_days))} 100.00% "
        f"({processed_days}/{len(open_days)} days), "
        f"daily_rows={total_daily_rows}, fundamental_rows={total_basic_rows}, "
        f"est_req_per_min={(processed_days * 3) / max(time.monotonic() - ingest_started, 1.0) * 60.0:.1f}"
    )
    if total_daily_rows == 0:
        raise RuntimeError("No daily bars fetched from Tushare for open days.")
    return {
        "status": "ok",
        "daily_rows": total_daily_rows,
        "fundamental_rows": total_basic_rows,
        "open_days": len(open_days),
    }


def _stage_clean_align(settings: Settings, engine: Engine) -> dict[str, object]:
    with engine.begin() as conn:
        frame = pd.read_sql(
            text(
                """
                SELECT
                  b.trade_date, b.ts_code, b.close, b.amount,
                  f.pe_ttm, f.pb, f.ps_ttm, f.dv_ttm
                FROM ods_daily_bar b
                JOIN meta_universe u
                  ON b.trade_date = u.trade_date
                 AND b.ts_code = u.ts_code
                 AND u.index_code = :index_code
                LEFT JOIN ods_fundamental f
                  ON b.trade_date = f.trade_date AND b.ts_code = f.ts_code
                WHERE b.trade_date BETWEEN :start_date AND :end_date
                """
            ),
            conn,
            params={
                "start_date": settings.run_start_date,
                "end_date": settings.run_end_date,
                "index_code": settings.universe_index,
            },
        )
    if frame.empty:
        raise RuntimeError("No ODS data available for clean_align.")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    feat = build_features(frame)
    out = feat[
        ["trade_date", "ts_code", "ret_1d", "ret_5d", "vol_20d", "mom_20d", "amt_ratio_20d", "pe_ttm", "pb", "ps_ttm", "dv_ttm", "is_valid"]
    ].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    rows = _upsert_dataframe(
        engine,
        "dwd_features_base",
        out,
        ["trade_date", "ts_code"],
        cleanup_where="trade_date BETWEEN :start_date AND :end_date",
        cleanup_params={"start_date": settings.run_start_date, "end_date": settings.run_end_date},
    )
    return {"status": "ok", "rows": rows}


def _stage_label_build(settings: Settings, engine: Engine) -> dict[str, object]:
    with engine.begin() as conn:
        bars = pd.read_sql(
            text(
                """
                SELECT b.trade_date, b.ts_code, b.close
                FROM ods_daily_bar b
                JOIN meta_universe u
                  ON b.trade_date = u.trade_date
                 AND b.ts_code = u.ts_code
                 AND u.index_code = :index_code
                WHERE b.trade_date BETWEEN :start_date AND :end_date
                """
            ),
            conn,
            params={
                "start_date": settings.run_start_date,
                "end_date": settings.run_end_date,
                "index_code": settings.universe_index,
            },
        )
    if bars.empty:
        raise RuntimeError("No daily bars available for label_build.")
    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    labeled = build_label_and_split(bars, settings.train_end_date, settings.valid_end_date)
    labels = labeled[["trade_date", "ts_code", "label_ret_t1"]].copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.date
    rows = _upsert_dataframe(
        engine,
        "dws_label",
        labels,
        ["trade_date", "ts_code"],
        cleanup_where="trade_date BETWEEN :start_date AND :end_date",
        cleanup_params={"start_date": settings.run_start_date, "end_date": settings.run_end_date},
    )
    return {"status": "ok", "rows": rows}


def _stage_dataset_split(settings: Settings, engine: Engine) -> dict[str, object]:
    with engine.begin() as conn:
        bars = pd.read_sql(
            text(
                """
                SELECT trade_date, ts_code, close
                FROM ods_daily_bar
                WHERE trade_date BETWEEN :start_date AND :end_date
                """
            ),
            conn,
            params={"start_date": settings.run_start_date, "end_date": settings.run_end_date},
        )
    if bars.empty:
        raise RuntimeError("No daily bars available for dataset_split.")
    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    labeled = build_label_and_split(bars, settings.train_end_date, settings.valid_end_date)
    split_df = labeled[["trade_date", "split_set"]].drop_duplicates(subset=["trade_date"], keep="last")
    split_df["trade_date"] = pd.to_datetime(split_df["trade_date"]).dt.date
    rows = _upsert_dataframe(
        engine,
        "ads_dataset_split",
        split_df,
        ["trade_date"],
        cleanup_where="trade_date BETWEEN :start_date AND :end_date",
        cleanup_params={"start_date": settings.run_start_date, "end_date": settings.run_end_date},
    )
    return {"status": "ok", "rows": rows}


def _stage_fit_predict(settings: Settings, engine: Engine, run_id: str) -> dict[str, object]:
    with engine.begin() as conn:
        dataset = pd.read_sql(
            text(
                """
                SELECT
                  f.trade_date, f.ts_code, f.ret_1d, f.ret_5d, f.vol_20d, f.mom_20d, f.amt_ratio_20d,
                  l.label_ret_t1, s.split_set
                FROM dwd_features_base f
                JOIN dws_label l ON f.trade_date = l.trade_date AND f.ts_code = l.ts_code
                JOIN ads_dataset_split s ON f.trade_date = s.trade_date
                WHERE f.trade_date BETWEEN :start_date AND :end_date
                """
            ),
            conn,
            params={"start_date": settings.run_start_date, "end_date": settings.run_end_date},
        )
    if dataset.empty:
        raise RuntimeError("No dataset rows found for fit_predict.")
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    feature_cols = ["ret_1d", "ret_5d", "vol_20d", "mom_20d", "amt_ratio_20d"]
    pred = fit_and_predict(dataset, feature_cols)
    pred = pred.copy()
    pred["run_id"] = run_id
    pred["model_name"] = "lightgbm"
    pred["model_version"] = "baseline-v1"
    pred["trade_date"] = pd.to_datetime(pred["trade_date"]).dt.date
    pred_out = pred[["run_id", "trade_date", "ts_code", "y_pred", "model_name", "model_version"]]
    rows = _upsert_dataframe(
        engine,
        "ads_pred_scores",
        pred_out,
        ["run_id", "trade_date", "ts_code"],
        cleanup_where="run_id = :run_id",
        cleanup_params={"run_id": run_id},
    )
    return {"status": "ok", "rows": rows}


def _stage_backtest(settings: Settings, engine: Engine, run_id: str) -> dict[str, object]:
    with engine.begin() as conn:
        pred = pd.read_sql(
            text(
                """
                SELECT trade_date, ts_code, y_pred
                FROM ads_pred_scores
                WHERE run_id = :run_id
                ORDER BY trade_date, ts_code
                """
            ),
            conn,
            params={"run_id": run_id},
        )
        realized = pd.read_sql(
            text(
                """
                SELECT trade_date, ts_code, label_ret_t1
                FROM dws_label
                WHERE trade_date BETWEEN :start_date AND :end_date
                """
            ),
            conn,
            params={"start_date": settings.run_start_date, "end_date": settings.run_end_date},
        )
    if pred.empty:
        raise RuntimeError("No predictions found for this run_id.")

    pred["trade_date"] = pd.to_datetime(pred["trade_date"])
    realized["trade_date"] = pd.to_datetime(realized["trade_date"])
    nav, metrics = topk_backtest(pred, realized, settings.top_k, settings.trade_cost_bps)

    nav_out = nav.copy()
    nav_out["trade_date"] = pd.to_datetime(nav_out["trade_date"]).dt.date
    nav_out["run_id"] = run_id
    nav_out = nav_out[["run_id", "trade_date", "nav"]]
    nav_rows = _upsert_dataframe(
        engine,
        "ads_backtest_nav",
        nav_out,
        ["run_id", "trade_date"],
        cleanup_where="run_id = :run_id",
        cleanup_params={"run_id": run_id},
    )

    metrics_df = pd.DataFrame(
        [{"run_id": run_id, "metric_name": k, "metric_value": float(v)} for k, v in metrics.items()]
    )
    metric_rows = _upsert_dataframe(
        engine,
        "ads_backtest_metrics",
        metrics_df,
        ["run_id", "metric_name"],
        cleanup_where="run_id = :run_id",
        cleanup_params={"run_id": run_id},
    )
    return {"status": "ok", "nav_rows": nav_rows, "metric_rows": metric_rows}


def _stage_evaluate_report(settings: Settings, engine: Engine, run_id: str) -> dict[str, object]:
    with engine.begin() as conn:
        nav = pd.read_sql(
            text(
                """
                SELECT trade_date, nav
                FROM ads_backtest_nav
                WHERE run_id = :run_id
                ORDER BY trade_date
                """
            ),
            conn,
            params={"run_id": run_id},
        )
    if nav.empty:
        raise RuntimeError("No nav rows found for evaluate_report.")

    report_dir = Path(settings.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"quantstats-{run_id}.html"

    try:
        import quantstats as qs
    except ImportError as exc:
        raise RuntimeError("quantstats is required for evaluate_report stage.") from exc

    nav["trade_date"] = pd.to_datetime(nav["trade_date"])
    nav = nav.sort_values("trade_date")
    returns = nav["nav"].pct_change().fillna(0.0)
    returns.index = nav["trade_date"]
    qs.reports.html(returns, output=str(report_path), title=f"Quant Report {run_id}")
    return {"status": "ok", "report_path": str(report_path)}


def _run_stage(stage_name: str, run_id: str, settings: Settings, engine: Engine) -> dict[str, object]:
    if stage_name == "universe_snapshot":
        return _stage_universe_snapshot(settings, engine)
    if stage_name == "raw_ingest":
        return _stage_raw_ingest(settings, engine)
    if stage_name == "clean_align":
        return _stage_clean_align(settings, engine)
    if stage_name == "label_build":
        return _stage_label_build(settings, engine)
    if stage_name == "dataset_split":
        return _stage_dataset_split(settings, engine)
    if stage_name == "fit_predict":
        return _stage_fit_predict(settings, engine, run_id)
    if stage_name == "backtest":
        return _stage_backtest(settings, engine, run_id)
    if stage_name == "evaluate_report":
        return _stage_evaluate_report(settings, engine, run_id)
    raise ValueError(f"Unknown stage: {stage_name}")


def run_pipeline(start_stage: str | None = None, execute: bool = False) -> dict[str, object]:
    if start_stage is None:
        start_index = 0
    else:
        if start_stage not in STAGES:
            raise ValueError(f"Unknown start_stage: {start_stage}")
        start_index = STAGES.index(start_stage)

    run_id = _make_run_id()

    if not execute:
        return {
            "run_id": run_id,
            "start_stage": STAGES[start_index],
            "stages": {stage_name: {"status": "ok"} for stage_name in STAGES[start_index:]},
        }

    settings = Settings.from_env()
    init_db(settings.mysql_dsn)
    engine = make_engine(settings.mysql_dsn)

    stage_results: dict[str, dict[str, object]] = {}
    for stage_name in STAGES[start_index:]:
        started_at = datetime.now(timezone.utc)
        _log(f"stage start: {stage_name} (run_id={run_id})")
        _log_stage(engine, run_id, stage_name, "running", started_at)
        try:
            payload = _run_stage(stage_name, run_id, settings, engine)
            finished_at = datetime.now(timezone.utc)
            _log_stage(engine, run_id, stage_name, "success", started_at, finished_at)
            _log(f"stage success: {stage_name} (run_id={run_id}) payload={payload}")
            stage_results[stage_name] = payload
        except Exception as exc:  # noqa: BLE001
            finished_at = datetime.now(timezone.utc)
            _log_stage(engine, run_id, stage_name, "failed", started_at, finished_at, str(exc))
            _log(f"stage failed: {stage_name} (run_id={run_id}) error={exc}")
            stage_results[stage_name] = {"status": "error", "error": str(exc)}
            raise

    return {
        "run_id": run_id,
        "start_stage": STAGES[start_index],
        "stages": stage_results,
    }
