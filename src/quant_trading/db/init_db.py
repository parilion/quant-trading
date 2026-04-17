from pathlib import Path

from sqlalchemy import text

from quant_trading.db.engine import make_engine


def init_db(mysql_dsn: str) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    schema_sql = schema_path.read_text(encoding="utf-8")
    engine = make_engine(mysql_dsn)
    statements = [stmt.strip() for stmt in schema_sql.split(";") if stmt.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
