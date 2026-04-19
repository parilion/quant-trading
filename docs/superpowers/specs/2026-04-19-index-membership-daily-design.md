# 2026-04-19 Index Membership Daily Table Design

## 1. Background and Problem

Current pipeline has a survivorship/look-ahead risk around index membership filtering:
- `meta_universe` is snapshot-like (sparse dates), not guaranteed daily coverage.
- Equality join (`b.trade_date = u.trade_date`) can over-filter rows and cause training-set collapse.
- Need a stable, auditable way to answer: "Was stock X a member of index Y on trade date T?"

Goal:
- Remove future-constituent leakage.
- Preserve enough daily samples for model training/backtest.
- Keep query logic simple and reproducible.

## 2. Scope

In scope:
- Add a daily membership table for index constituents.
- Add a new pipeline stage to expand snapshot membership into daily membership.
- Make `clean_align` and `label_build` consume the new daily table.
- Keep current workflow bounded to runtime window (`RUN_START_DATE`..`RUN_END_DATE`) first.

Out of scope:
- Full-history backfill in this iteration.
- Multi-index orchestration beyond configured `UNIVERSE_INDEX`.
- Strategy logic/model architecture changes.

## 3. Chosen Approach

Chosen: **Approach A: materialized daily membership table + pipeline consumption**.

Why:
- Prevents future leakage with explicit day-level membership semantics.
- Avoids expensive repeated interval computations in downstream queries.
- Easier debugging and auditing versus dynamic SQL-only expansion.

## 4. Data Model

New table: `dim_index_members_daily`

Columns:
- `trade_date DATE NOT NULL`
- `index_code VARCHAR(16) NOT NULL`
- `ts_code VARCHAR(16) NOT NULL`
- `source VARCHAR(32) NOT NULL DEFAULT 'meta_universe_expand'`
- `ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP`
- `update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP`

Keys and indexes:
- `PRIMARY KEY (trade_date, index_code, ts_code)`
- `KEY idx_index_trade (index_code, trade_date)`

Storage policy:
- Store only positive membership rows (equivalent to `is_member=1`).
- Do not store non-membership (`0`) rows.

## 5. Pipeline Changes

### 5.1 Stage order

Insert new stage `universe_daily_expand`:
1. `universe_snapshot`
2. `universe_daily_expand`  <-- new
3. `raw_ingest`
4. `clean_align`
5. `label_build`
6. `dataset_split`
7. `fit_predict`
8. `backtest`
9. `evaluate_report`

### 5.2 New stage: `universe_daily_expand`

Input:
- `meta_universe` snapshot rows for `UNIVERSE_INDEX`.
- Exchange trading calendar (`SSE`) for `RUN_START_DATE`..`RUN_END_DATE`.

Logic:
- For each `(index_code, ts_code)`, sort snapshot dates.
- Build effective intervals:
  - `effective_start = snapshot_date`
  - `effective_end = next_snapshot_date - 1 day`
  - Final interval capped by `RUN_END_DATE`.
- Intersect intervals with trading calendar days only.
- Write expanded daily rows into `dim_index_members_daily`.

Idempotency:
- Before insert, cleanup target range for the same index:
  - `index_code = :index_code AND trade_date BETWEEN :start_date AND :end_date`
- Then append regenerated rows.

### 5.3 Downstream consumption

`clean_align` and `label_build` must join `dim_index_members_daily` (not `meta_universe`):
- `b.trade_date = d.trade_date`
- `b.ts_code = d.ts_code`
- `d.index_code = :index_code`

## 6. Boundary Decisions

- If `RUN_START_DATE` is earlier than first available snapshot date, **do not backfill before first snapshot**.
- This is conservative and avoids inferred pre-membership.
- Resulting pre-snapshot window may have no eligible rows by design.

## 7. Error Handling and Observability

`universe_daily_expand` should fail fast when:
- No snapshot rows exist for configured index/range.
- No trading days exist in the runtime window.
- Expansion generates zero rows after applying conservative boundary policy.

Stage payload should include at least:
- `expanded_rows`
- `snapshot_dates_count`
- `trade_days_count`

## 8. Testing Strategy

Unit tests:
- Interval expansion correctness for multi-snapshot symbols.
- End-interval cap at `RUN_END_DATE`.
- Conservative behavior before first snapshot date.
- Idempotent cleanup/write behavior.

Integration tests:
- New stage appears in `STAGES` in correct order.
- `clean_align` and `label_build` queries use `dim_index_members_daily`.
- End-to-end run from `clean_align` has non-empty `train` split when date coverage supports it.

## 9. Success Criteria

Functional:
- "Stock is member on day T" can be answered directly from `dim_index_members_daily`.
- `clean_align`/`label_build` no longer depend on sparse snapshot equality join.

Quality:
- No future-constituent leakage from global distinct constituent sets.
- Training no longer fails due only to sparse membership snapshots (when window overlaps expanded membership coverage).

Operational:
- Pipeline remains rerunnable (idempotent for target window).
- Test suite passes for updated logic.

## 10. Rollout Plan

Phase 1 (this iteration):
- Implement schema + stage + downstream query switch for runtime window.

Phase 2 (later):
- Add optional full-history backfill mode.
- Add data-quality checks and automated drift alerts.
