from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    mysql_dsn: str
    tushare_token: str
    tushare_base_url: str = "http://tsy.xiaodefa.cn"
    universe_index: str = "000905.SH"
    top_k: int = 50
    trade_cost_bps: int = 20
    run_start_date: str = "2020-01-01"
    run_end_date: str = "2025-12-31"
    train_end_date: str = "2022-12-31"
    valid_end_date: str = "2023-12-31"
    report_dir: str = "artifacts/reports"

    @classmethod
    def from_env(cls) -> "Settings":
        top_k = cls._parse_int_env("TOP_K", 50)
        trade_cost_bps = cls._parse_int_env("TRADE_COST_BPS", 20)

        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if trade_cost_bps < 0:
            raise ValueError("trade_cost_bps must be >= 0")

        return cls(
            mysql_dsn=cls._get_required_env("MYSQL_DSN"),
            tushare_token=cls._get_required_env("TUSHARE_TOKEN"),
            tushare_base_url=os.environ.get("TUSHARE_BASE_URL", "http://tsy.xiaodefa.cn"),
            universe_index=os.environ.get("UNIVERSE_INDEX", "000905.SH"),
            top_k=top_k,
            trade_cost_bps=trade_cost_bps,
            run_start_date=os.environ.get("RUN_START_DATE", "2020-01-01"),
            run_end_date=os.environ.get("RUN_END_DATE", "2025-12-31"),
            train_end_date=os.environ.get("TRAIN_END_DATE", "2022-12-31"),
            valid_end_date=os.environ.get("VALID_END_DATE", "2023-12-31"),
            report_dir=os.environ.get("REPORT_DIR", "artifacts/reports"),
        )

    @staticmethod
    def _get_required_env(field: str) -> str:
        value = os.environ.get(field)
        if value is None or value == "":
            raise ValueError(f"Missing required environment variable: {field}")
        return value

    @staticmethod
    def _parse_int_env(field: str, default: int) -> int:
        raw = os.environ.get(field, str(default))
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid integer for {field}: {raw!r}") from exc
