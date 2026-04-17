from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    mysql_dsn: str
    tushare_token: str
    universe_index: str = "000905.SH"
    top_k: int = 50
    trade_cost_bps: int = 20

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
            universe_index=os.environ.get("UNIVERSE_INDEX", "000905.SH"),
            top_k=top_k,
            trade_cost_bps=trade_cost_bps,
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
