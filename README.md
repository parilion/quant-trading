# Quant Trading

## Quick Start

1. Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

2. Configure

```powershell
Copy-Item .env.example .env
```

Required configuration:
- `MYSQL_DSN`
- `TUSHARE_TOKEN`

Common runtime configuration:
- `UNIVERSE_INDEX` (default `000905.SH`)
- `TOP_K` (default `50`)
- `TRADE_COST_BPS` (default `20`)
- `RUN_START_DATE`, `RUN_END_DATE`
- `TRAIN_END_DATE`, `VALID_END_DATE`
- `REPORT_DIR`

3. Run pipeline

Dry run (no external IO, test-friendly):

```powershell
qt
```

Execute real stages (Tushare + MySQL):

```powershell
qt --execute
```

Start from a specific stage:

```powershell
qt --execute --start-stage fit_predict
```
