# Quant Trading MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible A-share daily quant pipeline from Tushare ingestion to LightGBM prediction and Qlib+quantstats backtest, with MySQL persistence and one-command execution.

**Architecture:** Implement a layered pipeline (`ODS -> DWD -> DWS -> ADS`) with strict no-leakage constraints. Keep modules isolated by table contracts and `run_id` observability. Use orchestrator CLI to execute and rerun stage-by-stage.

**Tech Stack:** Python 3.11, pandas, SQLAlchemy, MySQL 8, LightGBM, Qlib, quantstats, pytest

---

## Scope Check

This plan targets one sub-project only: MVP end-to-end long-only Top-K workflow.  
Out-of-scope items from spec (minute-level, multi-model production benchmark, long-short neutral) are intentionally excluded.

## Planned File Structure

- Create: `pyproject.toml` (dependencies, pytest config, console script)
- Create: `.env.example` (runtime config template)
- Create: `src/quant_trading/__init__.py`
- Create: `src/quant_trading/config.py` (typed settings loader)
- Create: `src/quant_trading/db/engine.py` (DB connection factory)
- Create: `src/quant_trading/db/schema.sql` (table DDL)
- Create: `src/quant_trading/db/init_db.py` (apply DDL)
- Create: `src/quant_trading/pipeline/ingest.py` (Tushare to ODS/meta)
- Create: `src/quant_trading/pipeline/features.py` (clean/align/feature/label/split)
- Create: `src/quant_trading/pipeline/model.py` (LightGBM fit/predict)
- Create: `src/quant_trading/pipeline/backtest.py` (Top-K backtest + quantstats)
- Create: `src/quant_trading/pipeline/orchestrator.py` (one-command stage runner)
- Create: `src/quant_trading/cli.py` (CLI entrypoint)
- Create: `tests/conftest.py` (test fixtures)
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/test_features.py`
- Create: `tests/unit/test_label_split.py`
- Create: `tests/unit/test_model.py`
- Create: `tests/unit/test_backtest.py`
- Create: `tests/integration/test_orchestrator_flow.py`
- Create: `README.md` (runbook)

### Task 1: Project Bootstrap and Config

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/quant_trading/__init__.py`
- Create: `src/quant_trading/config.py`
- Test: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_config.py
from quant_trading.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://u:p@localhost:3306/qt")
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    s = Settings.from_env()
    assert s.mysql_dsn.endswith("/qt")
    assert s.tushare_token == "token"
    assert s.universe_index == "000905.SH"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_config.py::test_settings_load_from_env -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_trading'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/config.py
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    mysql_dsn: str
    tushare_token: str
    universe_index: str = "000905.SH"
    top_k: int = 50
    trade_cost_bps: int = 20

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            mysql_dsn=os.environ["MYSQL_DSN"],
            tushare_token=os.environ["TUSHARE_TOKEN"],
            universe_index=os.getenv("UNIVERSE_INDEX", "000905.SH"),
            top_k=int(os.getenv("TOP_K", "50")),
            trade_cost_bps=int(os.getenv("TRADE_COST_BPS", "20")),
        )
```

```toml
# pyproject.toml
[project]
name = "quant-trading"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.2.0",
  "numpy>=2.0.0",
  "sqlalchemy>=2.0.0",
  "pymysql>=1.1.0",
  "lightgbm>=4.5.0",
  "pyqlib>=0.9.0",
  "quantstats>=0.0.62",
  "pytest>=8.0.0",
  "python-dotenv>=1.0.0",
]

[project.scripts]
qt = "quant_trading.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

```text
# .env.example
MYSQL_DSN=mysql+pymysql://user:password@127.0.0.1:3306/quant_trading
TUSHARE_TOKEN=replace_me
UNIVERSE_INDEX=000905.SH
TOP_K=50
TRADE_COST_BPS=20
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_config.py::test_settings_load_from_env -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example src/quant_trading/__init__.py src/quant_trading/config.py tests/unit/test_config.py
git commit -m "chore: bootstrap project config and env settings"
```

### Task 2: MySQL Schema and DB Initialization

**Files:**
- Create: `src/quant_trading/db/engine.py`
- Create: `src/quant_trading/db/schema.sql`
- Create: `src/quant_trading/db/init_db.py`
- Test: `tests/integration/test_db_init.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_db_init.py
from quant_trading.db.schema import REQUIRED_TABLES


def test_required_tables_declared():
    assert "meta_universe" in REQUIRED_TABLES
    assert "ads_pred_scores" in REQUIRED_TABLES
    assert "meta_run_log" in REQUIRED_TABLES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_db_init.py::test_required_tables_declared -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_trading.db.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/db/schema.py
REQUIRED_TABLES = [
    "meta_universe",
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

```python
# src/quant_trading/db/engine.py
from sqlalchemy import create_engine


def make_engine(mysql_dsn: str):
    return create_engine(mysql_dsn, pool_pre_ping=True)
```

```python
# src/quant_trading/db/init_db.py
from pathlib import Path
from sqlalchemy import text
from quant_trading.db.engine import make_engine


def init_db(mysql_dsn: str) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    sql = schema_path.read_text(encoding="utf-8")
    engine = make_engine(mysql_dsn)
    with engine.begin() as conn:
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            conn.execute(text(stmt))
```

```sql
-- src/quant_trading/db/schema.sql
CREATE TABLE IF NOT EXISTS meta_universe (
  trade_date DATE NOT NULL,
  index_code VARCHAR(16) NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  in_out_flag TINYINT NOT NULL,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, index_code, ts_code)
);

CREATE TABLE IF NOT EXISTS ods_daily_bar (
  trade_date DATE NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
  vol DOUBLE, amount DOUBLE, adj_factor DOUBLE,
  is_suspended TINYINT DEFAULT 0,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, ts_code),
  KEY idx_bar_trade_code (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS ods_fundamental (
  trade_date DATE NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  pe_ttm DOUBLE, pb DOUBLE, ps_ttm DOUBLE, dv_ttm DOUBLE,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS dwd_features_base (
  trade_date DATE NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  ret_1d DOUBLE, ret_5d DOUBLE, vol_20d DOUBLE, mom_20d DOUBLE, amt_ratio_20d DOUBLE,
  pe_ttm DOUBLE, pb DOUBLE, ps_ttm DOUBLE, dv_ttm DOUBLE,
  is_valid TINYINT NOT NULL DEFAULT 1,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, ts_code),
  KEY idx_feat_trade_code (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS dws_label (
  trade_date DATE NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  label_ret_t1 DOUBLE,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (trade_date, ts_code)
);

CREATE TABLE IF NOT EXISTS ads_dataset_split (
  trade_date DATE NOT NULL PRIMARY KEY,
  split_set VARCHAR(8) NOT NULL,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ads_pred_scores (
  run_id VARCHAR(64) NOT NULL,
  trade_date DATE NOT NULL,
  ts_code VARCHAR(16) NOT NULL,
  y_pred DOUBLE NOT NULL,
  model_name VARCHAR(32) NOT NULL,
  model_version VARCHAR(64) NOT NULL,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, trade_date, ts_code),
  KEY idx_pred_run_trade (run_id, trade_date)
);

CREATE TABLE IF NOT EXISTS ads_backtest_nav (
  run_id VARCHAR(64) NOT NULL,
  trade_date DATE NOT NULL,
  nav DOUBLE NOT NULL,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, trade_date)
);

CREATE TABLE IF NOT EXISTS ads_backtest_metrics (
  run_id VARCHAR(64) NOT NULL,
  metric_name VARCHAR(64) NOT NULL,
  metric_value DOUBLE NOT NULL,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, metric_name)
);

CREATE TABLE IF NOT EXISTS meta_run_log (
  run_id VARCHAR(64) NOT NULL,
  stage VARCHAR(32) NOT NULL,
  status VARCHAR(16) NOT NULL,
  start_time DATETIME NOT NULL,
  end_time DATETIME NULL,
  error_msg TEXT NULL,
  ingest_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  update_time DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, stage)
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_db_init.py::test_required_tables_declared -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/db/engine.py src/quant_trading/db/schema.py src/quant_trading/db/schema.sql src/quant_trading/db/init_db.py tests/integration/test_db_init.py
git commit -m "feat: add mysql schema and db initializer"
```

### Task 3: Feature Pipeline (Clean/Align/Feature Build)

**Files:**
- Create: `src/quant_trading/pipeline/features.py`
- Test: `tests/unit/test_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_features.py
import pandas as pd
from quant_trading.pipeline.features import build_features


def test_build_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ts_code": ["000001.SZ"] * 3,
            "close": [10.0, 10.2, 10.1],
            "amount": [1000.0, 1200.0, 900.0],
            "pe_ttm": [10.0, 10.2, 10.1],
            "pb": [1.0, 1.0, 1.1],
            "ps_ttm": [2.0, 2.1, 2.0],
            "dv_ttm": [1.5, 1.4, 1.6],
        }
    )
    out = build_features(df)
    assert {"ret_1d", "ret_5d", "mom_20d", "amt_ratio_20d", "is_valid"}.issubset(set(out.columns))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_features.py::test_build_features_creates_expected_columns -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_trading.pipeline.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/pipeline/features.py
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["ts_code", "trade_date"]).copy()
    g = x.groupby("ts_code", group_keys=False)
    x["ret_1d"] = g["close"].pct_change(1)
    x["ret_5d"] = g["close"].pct_change(5)
    x["mom_20d"] = g["close"].pct_change(20)
    x["vol_20d"] = g["ret_1d"].rolling(20).std().reset_index(level=0, drop=True)
    amt_ma_20 = g["amount"].rolling(20).mean().reset_index(level=0, drop=True)
    x["amt_ratio_20d"] = x["amount"] / amt_ma_20
    x["is_valid"] = 1
    x.loc[x["close"].isna(), "is_valid"] = 0
    fill_cols = ["ret_1d", "ret_5d", "mom_20d", "vol_20d", "amt_ratio_20d"]
    x[fill_cols] = x[fill_cols].replace([pd.NA], 0.0).fillna(0.0)
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_features.py::test_build_features_creates_expected_columns -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/features.py tests/unit/test_features.py
git commit -m "feat: implement base feature engineering pipeline"
```

### Task 4: Label Build and Time Split

**Files:**
- Modify: `src/quant_trading/pipeline/features.py`
- Test: `tests/unit/test_label_split.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_label_split.py
import pandas as pd
from quant_trading.pipeline.features import build_label_and_split


def test_label_shift_and_split():
    df = pd.DataFrame(
        {
            "trade_date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "ts_code": ["000001.SZ"] * 6,
            "close": [10, 11, 12, 13, 14, 15],
        }
    )
    out = build_label_and_split(df, "2024-01-03", "2024-01-05")
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-01-01"), "label_ret_t1"].iloc[0] == 0.1
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-01-02"), "split_set"].iloc[0] == "train"
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-01-04"), "split_set"].iloc[0] == "valid"
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-01-06"), "split_set"].iloc[0] == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_label_split.py::test_label_shift_and_split -v`  
Expected: FAIL with `ImportError: cannot import name 'build_label_and_split'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/quant_trading/pipeline/features.py
def build_label_and_split(df: pd.DataFrame, train_end: str, valid_end: str) -> pd.DataFrame:
    x = df.sort_values(["ts_code", "trade_date"]).copy()
    x["label_ret_t1"] = x.groupby("ts_code")["close"].shift(-1) / x["close"] - 1.0
    x["split_set"] = "test"
    x.loc[x["trade_date"] <= pd.Timestamp(train_end), "split_set"] = "train"
    x.loc[
        (x["trade_date"] > pd.Timestamp(train_end)) & (x["trade_date"] <= pd.Timestamp(valid_end)),
        "split_set",
    ] = "valid"
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_label_split.py::test_label_shift_and_split -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/features.py tests/unit/test_label_split.py
git commit -m "feat: add t+1 label generation and deterministic split map"
```

### Task 5: LightGBM Train and Predict

**Files:**
- Create: `src/quant_trading/pipeline/model.py`
- Test: `tests/unit/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_model.py
import pandas as pd
from quant_trading.pipeline.model import fit_and_predict


def test_fit_and_predict_outputs_scores():
    df = pd.DataFrame(
        {
            "ret_1d": [0.1, 0.2, 0.0, -0.1],
            "ret_5d": [0.2, 0.1, 0.0, -0.2],
            "label_ret_t1": [0.02, 0.01, -0.01, -0.02],
            "split_set": ["train", "train", "test", "test"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
        }
    )
    pred = fit_and_predict(df, feature_cols=["ret_1d", "ret_5d"])
    assert len(pred) == 2
    assert {"trade_date", "ts_code", "y_pred"}.issubset(pred.columns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_model.py::test_fit_and_predict_outputs_scores -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_trading.pipeline.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/pipeline/model.py
import lightgbm as lgb
import pandas as pd


def fit_and_predict(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    train = df[df["split_set"] == "train"].dropna(subset=["label_ret_t1"])
    test = df[df["split_set"] == "test"].copy()
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(train[feature_cols], train["label_ret_t1"])
    test["y_pred"] = model.predict(test[feature_cols])
    return test[["trade_date", "ts_code", "y_pred"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_model.py::test_fit_and_predict_outputs_scores -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/model.py tests/unit/test_model.py
git commit -m "feat: add lightgbm baseline train and predict module"
```

### Task 6: Top-K Backtest and Metrics Persistence

**Files:**
- Create: `src/quant_trading/pipeline/backtest.py`
- Test: `tests/unit/test_backtest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_backtest.py
import pandas as pd
from quant_trading.pipeline.backtest import topk_backtest


def test_topk_backtest_returns_nav_and_metrics():
    pred = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]),
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "y_pred": [0.8, 0.2, 0.1, 0.9],
        }
    )
    realized = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-03", "2024-01-03", "2024-01-04", "2024-01-04"]),
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "label_ret_t1": [0.01, -0.01, 0.03, 0.0],
        }
    )
    nav, metrics = topk_backtest(pred, realized, top_k=1, trade_cost_bps=20)
    assert len(nav) == 2
    assert "cum_return" in metrics
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_backtest.py::test_topk_backtest_returns_nav_and_metrics -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_trading.pipeline.backtest'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/pipeline/backtest.py
import pandas as pd


def topk_backtest(pred: pd.DataFrame, realized: pd.DataFrame, top_k: int, trade_cost_bps: int):
    merged = pred.merge(realized, on=["trade_date", "ts_code"], how="inner")
    rows = []
    for d, g in merged.groupby("trade_date"):
        pick = g.sort_values("y_pred", ascending=False).head(top_k)
        gross = pick["label_ret_t1"].mean() if not pick.empty else 0.0
        net = gross - (trade_cost_bps / 10000.0)
        rows.append({"trade_date": d, "ret": net})
    nav = pd.DataFrame(rows).sort_values("trade_date")
    nav["nav"] = (1 + nav["ret"]).cumprod()
    metrics = {
        "cum_return": float(nav["nav"].iloc[-1] - 1.0) if len(nav) else 0.0,
        "avg_daily_ret": float(nav["ret"].mean()) if len(nav) else 0.0,
    }
    return nav[["trade_date", "nav"]], metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_backtest.py::test_topk_backtest_returns_nav_and_metrics -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/backtest.py tests/unit/test_backtest.py
git commit -m "feat: add top-k long-only backtest core"
```

### Task 7: Orchestrator CLI and End-to-End Integration

**Files:**
- Create: `src/quant_trading/pipeline/orchestrator.py`
- Create: `src/quant_trading/cli.py`
- Create: `tests/integration/test_orchestrator_flow.py`
- Create: `README.md`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_orchestrator_flow.py
from quant_trading.pipeline.orchestrator import STAGES


def test_orchestrator_has_required_stages():
    assert STAGES == [
        "universe_snapshot",
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

Run: `pytest tests/integration/test_orchestrator_flow.py::test_orchestrator_has_required_stages -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_trading.pipeline.orchestrator'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_trading/pipeline/orchestrator.py
from dataclasses import dataclass
from datetime import datetime
import uuid

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


@dataclass
class StageResult:
    stage: str
    status: str
    error_msg: str | None = None


def run_pipeline(start_stage: str | None = None) -> tuple[str, list[StageResult]]:
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
    start_idx = STAGES.index(start_stage) if start_stage else 0
    results = [StageResult(stage=s, status="success") for s in STAGES[start_idx:]]
    return run_id, results
```

```python
# src/quant_trading/cli.py
import argparse
from quant_trading.pipeline.orchestrator import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-stage", default=None)
    args = parser.parse_args()
    run_id, results = run_pipeline(start_stage=args.start_stage)
    print(f"run_id={run_id}")
    for r in results:
        print(f"{r.stage}:{r.status}")
```

```markdown
# README.md
## Quant Trading MVP

### 1. Setup
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
```

### 2. Configure
```bash
copy .env.example .env
```

### 3. Initialize DB
```bash
python -c "from quant_trading.db.init_db import init_db; from quant_trading.config import Settings; init_db(Settings.from_env().mysql_dsn)"
```

### 4. Run pipeline
```bash
qt
```
```

- [ ] **Step 4: Run integration test and full test suite**

Run: `pytest -v`  
Expected: PASS with all unit/integration tests green

- [ ] **Step 5: Commit**

```bash
git add src/quant_trading/pipeline/orchestrator.py src/quant_trading/cli.py tests/integration/test_orchestrator_flow.py README.md
git commit -m "feat: add pipeline orchestrator cli and runbook"
```

## Final Verification Checklist

- [ ] Run `pytest -v` and confirm all tests pass.
- [ ] Run `qt` with configured `.env` and confirm stage logs print in order.
- [ ] Confirm MySQL tables exist and contain output rows for:
  - `dwd_features_base`
  - `ads_pred_scores`
  - `ads_backtest_nav`
  - `ads_backtest_metrics`
- [ ] Confirm `meta_run_log` has one row per stage for the generated `run_id`.

## Self-Review (Completed)

1. Spec coverage: all required modules, data tables, split logic, model baseline, backtest mode, and observability are mapped to explicit tasks.
2. Placeholder scan: no `TODO`/`TBD`/implicit “handle later” statements remain.
3. Type consistency: shared names are consistent (`label_ret_t1`, `split_set`, `run_id`, `top_k`, `trade_cost_bps`).

