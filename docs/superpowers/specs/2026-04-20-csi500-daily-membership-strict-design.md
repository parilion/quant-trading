# 2026-04-20 CSI500 Daily Membership Strict Design

## 1. Goal

Ensure training/backtest always uses **same-day accurate CSI500 constituents** (`000905.SH`) with no future leakage and no historical-union drift.

Confirmed business rules:
- Same-day membership definition: use the latest snapshot date `S` where `S <= trade_date`.
- If `RUN_START_DATE` is earlier than first available snapshot date: **fail fast**.
- If mapped snapshot constituent count is not exactly 500: **fail fast**.
- If `RUN_END_DATE` is later than last snapshot date: allow forward-fill from last snapshot to `RUN_END_DATE`.

## 2. Current Problem

Current expansion is per `(index_code, ts_code)` interval carry-forward.  
Effect: symbols that appeared historically can persist to `RUN_END_DATE`, causing cumulative union behavior (daily member count > 500).

This breaks strict CSI500-universe semantics for model training and backtest.

## 3. Chosen Approach

Rebuild `dim_index_members_daily` with **snapshot-interval mapping by global snapshot dates**, not per-symbol timelines.

For ordered snapshot dates `S1 < S2 < ... < Sn`:
- For each trade day `d`:
  - pick effective snapshot `Si` where `Si <= d < S(i+1)` (or `Sn <= d <= RUN_END_DATE` for tail);
  - membership on `d` is exactly the 500 constituents from snapshot `Si`.

This preserves no-leakage and keeps daily cardinality fixed at 500 (except hard-fail conditions).

## 4. Data Flow and Component Changes

### 4.1 `_stage_universe_snapshot` (no schema change)

Keep using `index_weight` to populate `meta_universe` snapshot rows.

### 4.2 `expand_snapshot_membership` (core algorithm rewrite)

File: `src/quant_trading/pipeline/universe_membership.py`

Replace per-symbol expansion with:
1. Build snapshot-day constituent map:
   - key: `(index_code, snapshot_date)`
   - value: set/list of `ts_code`
2. Validate each snapshot-day has exactly 500 distinct `ts_code` for target index.
3. Build trade-day -> effective snapshot-day mapping from trading calendar.
4. Emit rows as cartesian of `trade_day` and effective snapshot constituents.

Output columns remain:
- `trade_date, index_code, ts_code, source`

### 4.3 `_stage_universe_daily_expand` (strict boundary checks)

File: `src/quant_trading/pipeline/orchestrator.py`

Add checks before expansion:
- If `run_start_date < first_snapshot_date`: raise `RuntimeError` with actionable message.
- If no snapshots for target index: existing fail remains.
- Preserve cleanup+replace write for target date range.

Post-expansion checks:
- Expanded output non-empty.
- Daily distinct constituent count == 500 for each day in output window; otherwise fail.

### 4.4 Downstream stages

No query-shape change required:
- `clean_align` and `label_build` continue joining `dim_index_members_daily`.
- `fit_predict` and `backtest` behavior remains unchanged, but now receives correct universe.

## 5. Error Handling

Fail-fast error cases:
- `RUN_START_DATE < first_snapshot_date`
- any snapshot-day for target index has distinct constituent count != 500
- any generated trade day has distinct constituent count != 500
- no open trading days in configured window
- expansion output empty

Error messages must include:
- index code,
- offending date/range,
- expected vs actual counts.

## 6. Testing Strategy

### Unit tests (`tests/unit/test_universe_membership.py`)

Add/replace tests to validate:
- Global snapshot-interval mapping semantics (`d` uses latest `S <= d`).
- No per-symbol persistence beyond effective snapshot interval.
- Tail forward-fill from last snapshot to `RUN_END_DATE`.
- Hard failure when snapshot-day count != 500.

### Integration tests

Update/add tests for:
- `universe_daily_expand` fails when `RUN_START_DATE` is before first snapshot.
- Expanded table daily count is exactly 500 for sampled range.
- `clean_align` and `label_build` still join `dim_index_members_daily` and produce non-empty datasets for valid windows.

### Data-quality guard queries (runtime/checklist)

Recommended post-run assertions:
- `COUNT(DISTINCT ts_code) = 500` per `trade_date` in `dim_index_members_daily` for target window.
- Date coverage equals trade calendar coverage in target window (after boundary rules).

## 7. Rollout and Compatibility

Rollout steps:
1. Merge algorithm + checks.
2. Re-run stages from `universe_daily_expand` onward.
3. Verify daily count and sample constituent dates.
4. Re-run training/backtest.

Compatibility:
- Table schema unchanged.
- Downstream SQL unchanged.
- Semantics corrected (strict daily CSI500 universe).

## 8. Success Criteria

Functional:
- For every trade date in runnable window, `dim_index_members_daily` for `000905.SH` has exactly 500 rows (distinct `ts_code`).
- Training and backtest are effectively constrained to same-day CSI500 constituents.

Quality:
- No future leakage.
- No cumulative historical-union drift.

Operational:
- Pipeline fails fast on invalid boundary/data-quality states with clear diagnostics.
