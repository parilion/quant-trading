# CSI500 Daily Membership Strict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make training/backtest consume exact same-day CSI500 constituents (500/day) via strict snapshot-to-trade-day mapping.

**Architecture:** Keep `meta_universe` as snapshot source and rebuild `dim_index_members_daily` using global snapshot intervals (`latest snapshot <= trade_date`). Add fail-fast checks in expansion and stage orchestration. Keep downstream joins unchanged so semantics are corrected at source.

**Tech Stack:** Python, pandas, SQLAlchemy, pytest, MySQL

---

## File Structure

- Modify: `src/quant_trading/pipeline/universe_membership.py`
  - Replace per-symbol interval carry-forward with global snapshot-interval mapping.
- Modify: `src/quant_trading/pipeline/orchestrator.py`
  - Add strict boundary checks and per-day `=500` validation in `universe_daily_expand`.
- Modify: `tests/unit/test_universe_membership.py`
  - Replace/add tests for strict interval semantics and cardinality checks.
- Add: `tests/unit/test_orchestrator_universe_daily_expand_strict.py`
  - Stage-level fail-fast and validation tests with monkeypatch.
- Modify: `AGENTS.md`
  - Update status after implementation/verification (remove stale anomaly diagnosis once fixed).

### Task 1: Lock strict semantics with failing unit tests

**Files:**
- Modify: `tests/unit/test_universe_membership.py`

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd
import pytest

from quant_trading.pipeline.universe_membership import expand_snapshot_membership


def _codes(n: int, prefix: str = "000") -> list[str]:
    return [f"{prefix}{i:03d}.SZ" for i in range(1, n + 1)]


def test_expand_snapshot_membership_uses_global_snapshot_interval_mapping():
    s1 = _codes(500, "100")
    s2 = _codes(500, "200")
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2024-01-31"] * 500 + ["2024-02-29"] * 500,
            "index_code": ["000905.SH"] * 1000,
            "ts_code": s1 + s2,
        }
    )
    trade_days = pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-29", "2024-03-01"])

    out = expand_snapshot_membership(snapshots, trade_days, "2024-02-01", "2024-03-01")

    d1 = out.loc[out["trade_date"] == pd.Timestamp("2024-02-01"), "ts_code"].nunique()
    d2 = out.loc[out["trade_date"] == pd.Timestamp("2024-02-29"), "ts_code"].nunique()
    assert d1 == 500 and d2 == 500
    assert set(out.loc[out["trade_date"] == pd.Timestamp("2024-02-01"), "ts_code"]) == set(s1)
    assert set(out.loc[out["trade_date"] == pd.Timestamp("2024-02-29"), "ts_code"]) == set(s2)


def test_expand_snapshot_membership_raises_when_snapshot_count_not_500():
    snapshots = pd.DataFrame(
        {
            "trade_date": ["2024-01-31"] * 499,
            "index_code": ["000905.SH"] * 499,
            "ts_code": _codes(499, "300"),
        }
    )
    trade_days = pd.to_datetime(["2024-02-01"])

    with pytest.raises(ValueError, match="exactly 500"):
        expand_snapshot_membership(snapshots, trade_days, "2024-02-01", "2024-02-01")
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/unit/test_universe_membership.py -q`  
Expected: FAIL due to current per-symbol carry-forward behavior.

- [ ] **Step 3: Commit test-only red state**

```bash
git add tests/unit/test_universe_membership.py
git commit -m "test: define strict daily CSI500 membership semantics"
```

### Task 2: Implement strict global snapshot interval expansion

**Files:**
- Modify: `src/quant_trading/pipeline/universe_membership.py`
- Test: `tests/unit/test_universe_membership.py`

- [ ] **Step 1: Implement minimal passing code**

```python
def expand_snapshot_membership(...):
    columns = ["trade_date", "index_code", "ts_code", "source"]
    if snapshots.empty:
        return pd.DataFrame(columns=columns)

    x = snapshots[["trade_date", "index_code", "ts_code"]].copy()
    x["trade_date"] = pd.to_datetime(x["trade_date"])
    x = x.drop_duplicates(["trade_date", "index_code", "ts_code"])

    run_start = pd.to_datetime(run_start_date)
    run_end = pd.to_datetime(run_end_date)
    trade_days = pd.to_datetime(pd.Series(trade_days)).drop_duplicates().sort_values()
    trade_days = trade_days[(trade_days >= run_start) & (trade_days <= run_end)]
    if trade_days.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for index_code, ix_df in x.groupby("index_code", sort=False):
        snap_days = sorted(ix_df["trade_date"].drop_duplicates().tolist())
        snap_map = {
            sd: set(ix_df.loc[ix_df["trade_date"] == sd, "ts_code"].tolist())
            for sd in snap_days
        }
        for sd, members in snap_map.items():
            if len(members) != 500:
                raise ValueError(f"Snapshot {index_code} {sd.date()} must contain exactly 500 members, got {len(members)}")

        for i, sd in enumerate(snap_days):
            next_sd = snap_days[i + 1] if i + 1 < len(snap_days) else None
            interval_start = max(sd, run_start)
            interval_end = min((next_sd - pd.Timedelta(days=1)) if next_sd is not None else run_end, run_end)
            if interval_start > interval_end:
                continue
            eff_days = trade_days[(trade_days >= interval_start) & (trade_days <= interval_end)]
            members = sorted(snap_map[sd])
            rows.extend((d, index_code, ts, "meta_universe_expand") for d in eff_days for ts in members)

    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    return out.drop_duplicates(["trade_date", "index_code", "ts_code"]).sort_values(
        ["trade_date", "index_code", "ts_code"]
    ).reset_index(drop=True)
```

- [ ] **Step 2: Run unit tests**

Run: `pytest tests/unit/test_universe_membership.py -q`  
Expected: PASS.

- [ ] **Step 3: Commit implementation**

```bash
git add src/quant_trading/pipeline/universe_membership.py tests/unit/test_universe_membership.py
git commit -m "feat: strict snapshot-interval expansion for daily CSI500 membership"
```

### Task 3: Add fail-fast checks in `universe_daily_expand`

**Files:**
- Modify: `src/quant_trading/pipeline/orchestrator.py`
- Add: `tests/unit/test_orchestrator_universe_daily_expand_strict.py`

- [ ] **Step 1: Write failing stage-level tests**

```python
def test_universe_daily_expand_fails_when_run_start_before_first_snapshot(...):
    # monkeypatch pd.read_sql to return snapshots with first date 2020-01-31
    # settings.run_start_date='2020-01-01'
    # expect RuntimeError with "earlier than first snapshot"

def test_universe_daily_expand_fails_when_daily_count_not_500(...):
    # monkeypatch expand_snapshot_membership to return one day with !=500 members
    # expect RuntimeError with "exactly 500"
```

- [ ] **Step 2: Run failing tests**

Run: `pytest tests/unit/test_orchestrator_universe_daily_expand_strict.py -q`  
Expected: FAIL.

- [ ] **Step 3: Implement checks in orchestrator**

```python
first_snapshot = pd.to_datetime(snapshots["trade_date"]).min()
if pd.to_datetime(settings.run_start_date) < first_snapshot:
    raise RuntimeError(
        f"RUN_START_DATE {settings.run_start_date} is earlier than first snapshot "
        f"{first_snapshot.date()} for index {settings.universe_index}."
    )

# after expanded built
day_counts = expanded.groupby("trade_date")["ts_code"].nunique()
bad = day_counts[day_counts != 500]
if not bad.empty:
    d = bad.index[0]
    c = int(bad.iloc[0])
    raise RuntimeError(
        f"Daily membership must be exactly 500 for {settings.universe_index}; "
        f"got {c} on {pd.to_datetime(d).date()}."
    )
```

- [ ] **Step 4: Run stage-level tests**

Run: `pytest tests/unit/test_orchestrator_universe_daily_expand_strict.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/orchestrator.py tests/unit/test_orchestrator_universe_daily_expand_strict.py
git commit -m "feat: add fail-fast guards for strict daily CSI500 membership"
```

### Task 4: Regression verification on current test suite

**Files:**
- Modify: none (verification task)

- [ ] **Step 1: Run focused pipeline unit tests**

Run:
```bash
pytest tests/unit/test_universe_membership.py tests/unit/test_orchestrator_universe_filter.py tests/unit/test_orchestrator_universe_daily_expand_strict.py -q
```
Expected: PASS.

- [ ] **Step 2: Run broader quick suite**

Run: `pytest tests/unit -q`  
Expected: PASS (or only known unrelated pre-existing failures).

- [ ] **Step 3: Commit if test snapshot docs/log updates are needed**

```bash
git add -A
git commit -m "test: verify strict daily membership changes across unit suite"
```

### Task 5: Rebuild data and verify production semantics

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Re-run pipeline from daily expansion**

Run: `python -c "from quant_trading.cli import main; main(['--execute','--start-stage','universe_daily_expand'])"`  
Expected: stages `universe_daily_expand` onward complete.

- [ ] **Step 2: Validate DB cardinality**

Run (SQL):
```sql
SELECT trade_date, COUNT(DISTINCT ts_code) AS c
FROM quant_trading.dim_index_members_daily
WHERE index_code='000905.SH'
  AND trade_date BETWEEN '2020-01-02' AND '2026-04-17'
GROUP BY trade_date
HAVING c<>500
LIMIT 5;
```
Expected: empty result.

- [ ] **Step 3: Update AGENTS runtime status**

Document in `AGENTS.md`:
- new date range coverage,
- per-day cardinality check result,
- explicit statement that backtest now reads exact same-day CSI500 mapping.

- [ ] **Step 4: Commit runtime verification notes**

```bash
git add AGENTS.md
git commit -m "docs: record strict same-day CSI500 verification results"
```

## Self-Review Checklist

- Spec coverage:
  - Same-day latest snapshot mapping: Task 1/2
  - start-before-first-snapshot fail: Task 3
  - per-day exactly-500 fail: Task 2/3/5
  - allow tail forward-fill: Task 2 tests + implementation
- Placeholder scan: no TBD/TODO placeholders left.
- Type/signature consistency:
  - `expand_snapshot_membership(...)` signature unchanged.
  - stage API `_stage_universe_daily_expand` unchanged.

