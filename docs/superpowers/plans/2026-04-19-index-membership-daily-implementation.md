# Index Membership Daily Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a daily constituent membership table for CSI 500 usage, remove future-constituent leakage, and keep training data coverage stable.

**Architecture:** Introduce `dim_index_members_daily` as a materialized day-level membership table generated from `meta_universe` snapshots plus trading calendar days. Add a new pipeline stage `universe_daily_expand` to build this table per run window, then make `clean_align` and `label_build` join this table instead of snapshot equality joins.

**Tech Stack:** Python 3.11, pandas, SQLAlchemy, MySQL, pytest.

---

## File Structure and Responsibilities

- Modify: `src/quant_trading/db/schema.sql`
  - Add `dim_index_members_daily` table DDL.
- Modify: `src/quant_trading/db/schema.py`
  - Register new required table name.
- Modify: `tests/integration/test_db_init.py`
  - Update required-table expectation.
- Create: `src/quant_trading/pipeline/universe_membership.py`
  - Pure functions to expand snapshot memberships into daily memberships.
- Create: `tests/unit/test_universe_membership.py`
  - Unit tests for interval expansion and boundary behavior.
- Modify: `src/quant_trading/pipeline/orchestrator.py`
  - Add stage `universe_daily_expand`, call expansion logic, and switch downstream joins.
- Modify: `tests/integration/test_orchestrator_flow.py`
  - Validate stage order includes new stage.
- Modify: `tests/unit/test_orchestrator_universe_filter.py`
  - Assert `clean_align` and `label_build` join `dim_index_members_daily`.

---

### Task 1: Add Daily Membership Table Schema

**Files:**
- Modify: `src/quant_trading/db/schema.sql`
- Modify: `src/quant_trading/db/schema.py`
- Test: `tests/integration/test_db_init.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_db_init.py

def test_required_tables_match_expected_plan_tables():
    expected_tables = {
        "meta_universe",
        "dim_index_members_daily",
        "ods_daily_bar",
        "ods_fundamental",
        "dwd_features_base",
        "dws_label",
        "ads_dataset_split",
        "ads_pred_scores",
        "ads_backtest_nav",
        "ads_backtest_metrics",
        "meta_run_log",
    }
    assert set(REQUIRED_TABLES) == expected_tables
    assert len(REQUIRED_TABLES) == 11
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_db_init.py::test_required_tables_match_expected_plan_tables -q`
Expected: FAIL with missing `dim_index_members_daily`.

- [ ] **Step 3: Write minimal implementation**

```sql
-- src/quant_trading/db/schema.sql
CREATE TABLE IF NOT EXISTS dim_index_members_daily (
  trade_date DATE NOT NULL,
  index_code VARCHAR(16) NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  source VARCHAR(32) NOT NULL DEFAULT 'meta_universe_expand',
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, index_code, ts_code),
  KEY idx_index_trade (index_code, trade_date)
);
```

```python
# src/quant_trading/db/schema.py
REQUIRED_TABLES = [
    "meta_universe",
    "dim_index_members_daily",
    "ods_daily_bar",
    "ods_fundamental",
    "dwd_features_base",
    "dws_label",
    "ads_dataset_split",
    "ads_pred_scores",
    "ads_backtest_nav",
    "ads_backtest_metrics",
    "meta_run_log",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_db_init.py::test_required_tables_match_expected_plan_tables -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/db/schema.sql src/quant_trading/db/schema.py tests/integration/test_db_init.py
git commit -m "feat: add daily index membership schema"
```

---

### Task 2: Implement Snapshot-to-Daily Expansion Logic (Pure Functions)

**Files:**
- Create: `src/quant_trading/pipeline/universe_membership.py`
- Create: `tests/unit/test_universe_membership.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_universe_membership.py
import pandas as pd
from quant_trading.pipeline.universe_membership import expand_snapshot_membership


def test_expand_snapshot_membership_builds_daily_rows():
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2024-01-02", "2024-01-05", "2024-01-02"],
            "index_code": ["000905.SH", "000905.SH", "000905.SH"],
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
        }
    )
    trade_days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])

    out = expand_snapshot_membership(snapshots, trade_days, "2024-01-01", "2024-01-08")

    assert (out["index_code"] == "000905.SH").all()
    assert set(out.columns) == {"trade_date", "index_code", "ts_code", "source"}
    assert ((out["trade_date"] >= pd.Timestamp("2024-01-01")) & (out["trade_date"] <= pd.Timestamp("2024-01-08"))).all()


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_universe_membership.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing function.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/pipeline/universe_membership.py
from __future__ import annotations

import pandas as pd


def expand_snapshot_membership(
    snapshots: pd.DataFrame,
    trade_days: pd.Series | pd.DatetimeIndex,
    run_start_date: str,
    run_end_date: str,
) -> pd.DataFrame:
    if snapshots.empty:
        return pd.DataFrame(columns=["trade_date", "index_code", "ts_code", "source"])

    x = snapshots[["trade_date", "index_code", "ts_code"]].copy()
    x["trade_date"] = pd.to_datetime(x["trade_date"])
    x = x.drop_duplicates(subset=["trade_date", "index_code", "ts_code"]).sort_values(["index_code", "ts_code", "trade_date"])

    td = pd.to_datetime(pd.Series(trade_days).drop_duplicates()).sort_values()
    run_start = pd.to_datetime(run_start_date)
    run_end = pd.to_datetime(run_end_date)
    td = td[(td >= run_start) & (td <= run_end)]

    rows: list[tuple[pd.Timestamp, str, str, str]] = []

    for (index_code, ts_code), grp in x.groupby(["index_code", "ts_code"], sort=False):
        dates = grp["trade_date"].tolist()
        for i, start in enumerate(dates):
            next_start = dates[i + 1] if i + 1 < len(dates) else None
            end = (next_start - pd.Timedelta(days=1)) if next_start is not None else run_end
            start_eff = max(start, run_start)
            end_eff = min(end, run_end)
            if start_eff > end_eff:
                continue
            effective_days = td[(td >= start_eff) & (td <= end_eff)]
            rows.extend((d, index_code, ts_code, "meta_universe_expand") for d in effective_days)

    out = pd.DataFrame(rows, columns=["trade_date", "index_code", "ts_code", "source"])
    if out.empty:
        return out
    return out.drop_duplicates(subset=["trade_date", "index_code", "ts_code"]).sort_values(
        ["trade_date", "index_code", "ts_code"]
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_universe_membership.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/universe_membership.py tests/unit/test_universe_membership.py
git commit -m "feat: add snapshot-to-daily universe expansion logic"
```

---

### Task 3: Add `universe_daily_expand` Stage and Wire It into Pipeline

**Files:**
- Modify: `src/quant_trading/pipeline/orchestrator.py`
- Test: `tests/integration/test_orchestrator_flow.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_orchestrator_flow.py

def test_stages_match_expected_order():
    assert STAGES == [
        "universe_snapshot",
        "universe_daily_expand",
        "raw_ingest",
        "clean_align",
        "label_build",
        "dataset_split",
        "fit_predict",
        "backtest",
        "evaluate_report",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_orchestrator_flow.py::test_stages_match_expected_order -q`
Expected: FAIL because stage is missing.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/pipeline/orchestrator.py (imports)
from quant_trading.pipeline.universe_membership import expand_snapshot_membership
```

```python
# src/quant_trading/pipeline/orchestrator.py (STAGES)
STAGES = [
    "universe_snapshot",
    "universe_daily_expand",
    "raw_ingest",
    "clean_align",
    "label_build",
    "dataset_split",
    "fit_predict",
    "backtest",
    "evaluate_report",
]
```

```python
# src/quant_trading/pipeline/orchestrator.py (new function)
def _stage_universe_daily_expand(settings: Settings, engine: Engine) -> dict[str, object]:
    pro = _get_thread_tushare_client(settings)
    with engine.begin() as conn:
        snapshots = pd.read_sql(
            text(
                """
                SELECT trade_date, index_code, ts_code
                FROM meta_universe
                WHERE index_code = :index_code
                  AND trade_date <= :end_date
                """
            ),
            conn,
            params={"index_code": settings.universe_index, "end_date": settings.run_end_date},
        )
    if snapshots.empty:
        raise RuntimeError("meta_universe has no rows for universe_daily_expand.")

    trade_cal = _call_with_retry(
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=_to_tushare_date(settings.run_start_date),
            end_date=_to_tushare_date(settings.run_end_date),
        )
    )
    if trade_cal is None or trade_cal.empty:
        raise RuntimeError("No trading calendar fetched for universe_daily_expand.")

    open_days = pd.to_datetime(trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"].astype(str))
    if open_days.empty:
        raise RuntimeError("No open trading days in configured range for universe_daily_expand.")

    expanded = expand_snapshot_membership(
        snapshots=snapshots,
        trade_days=open_days,
        run_start_date=settings.run_start_date,
        run_end_date=settings.run_end_date,
    )
    if expanded.empty:
        raise RuntimeError("universe_daily_expand produced zero membership rows.")

    expanded["trade_date"] = pd.to_datetime(expanded["trade_date"]).dt.date
    rows = _upsert_dataframe(
        engine,
        "dim_index_members_daily",
        expanded,
        ["trade_date", "index_code", "ts_code"],
        cleanup_where="index_code = :index_code AND trade_date BETWEEN :start_date AND :end_date",
        cleanup_params={
            "index_code": settings.universe_index,
            "start_date": settings.run_start_date,
            "end_date": settings.run_end_date,
        },
    )
    return {
        "status": "ok",
        "expanded_rows": rows,
        "snapshot_dates_count": int(pd.to_datetime(snapshots["trade_date"]).nunique()),
        "trade_days_count": int(open_days.nunique()),
    }
```

```python
# src/quant_trading/pipeline/orchestrator.py (_run_stage)
if stage_name == "universe_daily_expand":
    return _stage_universe_daily_expand(settings, engine)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_orchestrator_flow.py::test_stages_match_expected_order -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/orchestrator.py tests/integration/test_orchestrator_flow.py
git commit -m "feat: add universe_daily_expand pipeline stage"
```

---

### Task 4: Switch `clean_align` and `label_build` to Daily Membership Table

**Files:**
- Modify: `src/quant_trading/pipeline/orchestrator.py`
- Modify: `tests/unit/test_orchestrator_universe_filter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_orchestrator_universe_filter.py

def test_stage_clean_align_filters_by_daily_universe(monkeypatch):
    ...
    sql_text = str(captured["sql"]).lower()
    assert "join dim_index_members_daily d" in sql_text
    assert "d.index_code = :index_code" in sql_text


def test_stage_label_build_filters_by_daily_universe(monkeypatch):
    ...
    sql_text = str(captured["sql"]).lower()
    assert "join dim_index_members_daily d" in sql_text
    assert "d.index_code = :index_code" in sql_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_orchestrator_universe_filter.py -q`
Expected: FAIL because query still references `meta_universe`.

- [ ] **Step 3: Write minimal implementation**

```sql
-- src/quant_trading/pipeline/orchestrator.py (_stage_clean_align SQL)
FROM ods_daily_bar b
JOIN dim_index_members_daily d
  ON b.trade_date = d.trade_date
 AND b.ts_code = d.ts_code
 AND d.index_code = :index_code
LEFT JOIN ods_fundamental f
  ON b.trade_date = f.trade_date AND b.ts_code = f.ts_code
```

```sql
-- src/quant_trading/pipeline/orchestrator.py (_stage_label_build SQL)
FROM ods_daily_bar b
JOIN dim_index_members_daily d
  ON b.trade_date = d.trade_date
 AND b.ts_code = d.ts_code
 AND d.index_code = :index_code
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_orchestrator_universe_filter.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/orchestrator.py tests/unit/test_orchestrator_universe_filter.py
git commit -m "fix: consume daily membership table in clean and label stages"
```

---

### Task 5: End-to-End Validation and Regression Safety

**Files:**
- Modify: `tests/integration/test_orchestrator_flow.py`
- Modify: `docs/superpowers/specs/2026-04-19-index-membership-daily-design.md` (only if behavior notes need sync)

- [ ] **Step 1: Write failing integration expectation for new stage in dry-run**

```python
# tests/integration/test_orchestrator_flow.py

def test_run_pipeline_returns_run_id_and_stage_results():
    result = run_pipeline()
    assert list(result["stages"].keys()) == STAGES
    assert "universe_daily_expand" in result["stages"]
```

- [ ] **Step 2: Run focused integration tests**

Run: `pytest tests/integration/test_orchestrator_flow.py -q`
Expected: PASS (or FAIL first if ordering changed but assertions not yet updated).

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/unit -q`
Expected: all PASS.

Run: `pytest tests/integration -q`
Expected: all PASS.

- [ ] **Step 4: Execute runtime window rebuild and verify train split exists**

Run:
```bash
python -m quant_trading.cli --execute --start-stage universe_daily_expand
python -m quant_trading.cli --execute --start-stage clean_align
```

Expected: second command reaches at least `fit_predict` without `training set is empty`.

- [ ] **Step 5: Verify membership quality with SQL checks**

Run:
```sql
SELECT COUNT(*)
FROM dwd_features_base f
LEFT JOIN dim_index_members_daily d
  ON f.trade_date=d.trade_date AND f.ts_code=d.ts_code AND d.index_code='000905.SH'
WHERE d.ts_code IS NULL;
```
Expected: `0`.

Run:
```sql
SELECT split_set, COUNT(*)
FROM dwd_features_base f
JOIN dws_label l ON f.trade_date=l.trade_date AND f.ts_code=l.ts_code
JOIN ads_dataset_split s ON f.trade_date=s.trade_date
GROUP BY split_set;
```
Expected: includes `train` rows and `test` rows.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_orchestrator_flow.py
git commit -m "test: validate daily membership stage wiring and regression coverage"
```

---

## Self-Review Checklist (Plan vs Spec)

- Spec coverage:
  - New table schema: Task 1.
  - New stage and stage order: Task 3.
  - Downstream consumption switch: Task 4.
  - Conservative boundary and training viability checks: Tasks 2 and 5.
- Placeholder scan: no `TODO/TBD/implement later` left in tasks.
- Type/signature consistency:
  - New function name is consistently `expand_snapshot_membership`.
  - New stage name is consistently `universe_daily_expand`.
  - New table name is consistently `dim_index_members_daily`.
