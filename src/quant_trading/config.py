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
        return cls(
            mysql_dsn=os.environ["MYSQL_DSN"],
            tushare_token=os.environ["TUSHARE_TOKEN"],
            universe_index=os.environ.get("UNIVERSE_INDEX", "000905.SH"),
            top_k=int(os.environ.get("TOP_K", 50)),
            trade_cost_bps=int(os.environ.get("TRADE_COST_BPS", 20)),
        )
