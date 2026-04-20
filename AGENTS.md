# AGENTS Context: CSI500 (000905.SH) Daily Membership Status

Last checked: 2026-04-20 (Asia/Shanghai)  
Checker: Codex + MySQL MCP (`127.0.0.1:3307`, multi-DB mode)

## Current Status (After Strict Fix)

Training/backtest now use strict same-day CSI500 membership semantics:
- each trade day maps to latest snapshot day `S` where `S <= trade_date`
- no future leakage
- no cumulative historical-union drift

## Core Tables

- `quant_trading.meta_universe` (snapshot membership source)
- `quant_trading.dim_index_members_daily` (daily membership used by clean/label stages)
- `quant_trading.ods_daily_bar` (market bars)

## Verified Metrics

### Snapshot Source: `meta_universe` (`index_code='000905.SH'`)

- Date range: `2015-01-30` to `2026-03-31`
- Total rows: `67,500`
- Snapshot days: `135`
- Snapshot cardinality: each snapshot day = `500` symbols

### Daily Membership: `dim_index_members_daily` (`index_code='000905.SH'`)

Validated window: `2020-01-02` to `2026-04-17`

- Total rows: `761,500`
- Distinct trade days: `1,523`
- Per-day cardinality check:
  - bad days (`COUNT(DISTINCT ts_code) <> 500`): `0`
  - sample checks:
    - `2020-01-02`: `500`
    - `2023-12-29`: `500`
    - `2026-04-17`: `500`

Interpretation: daily membership is now constrained to exact 500 constituents per trade day in validated window.

### ODS Bars: `ods_daily_bar`

- Date range: `2015-01-05` to `2026-04-17`
- Total rows: `1,359,885`
- Distinct trade days: `2,742`

## Boundary Rules Enforced in Code

- If `RUN_START_DATE` is earlier than first snapshot date for target index: fail fast.
- If any snapshot day or generated daily membership day has count != 500: fail fast.
- If `RUN_END_DATE` exceeds last snapshot date: last snapshot is allowed to forward-fill to `RUN_END_DATE`.

## Recheck SQL

```sql
SELECT DATE_FORMAT(MIN(trade_date),'%Y-%m-%d') AS min_date,
       DATE_FORMAT(MAX(trade_date),'%Y-%m-%d') AS max_date,
       COUNT(*) AS total_rows,
       COUNT(DISTINCT DATE(trade_date)) AS total_days
FROM quant_trading.meta_universe
WHERE index_code='000905.SH';
```

```sql
SELECT COUNT(*) AS bad_days
FROM (
  SELECT trade_date, COUNT(DISTINCT ts_code) AS c
  FROM quant_trading.dim_index_members_daily
  WHERE index_code='000905.SH'
    AND trade_date BETWEEN '2020-01-02' AND '2026-04-17'
  GROUP BY trade_date
  HAVING c<>500
) t;
```

```sql
SELECT DATE_FORMAT(trade_date,'%Y-%m-%d') AS trade_date, COUNT(DISTINCT ts_code) AS c
FROM quant_trading.dim_index_members_daily
WHERE index_code='000905.SH'
  AND trade_date IN ('2020-01-02','2023-12-29','2026-04-17')
GROUP BY trade_date
ORDER BY trade_date;
```
