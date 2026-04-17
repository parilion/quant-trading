# Quant Trading

## Minimal Runbook

1. Setup environment

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

2. Configure environment

```bash
copy .env.example .env
```

3. Initialize database schema

```bash
python -m quant_trading.db.init_db
```

4. Run full pipeline

```bash
qt
```

5. Run from a specific stage

```bash
qt --start-stage fit_predict
```
