from pathlib import Path

from sqlalchemy import text

from quant_trading.db.engine import make_engine


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    buffer: list[str] = []
    in_single_quote = False
    in_double_quote = False
    i = 0
    length = len(sql_text)

    while i < length:
        ch = sql_text[i]
        nxt = sql_text[i + 1] if i + 1 < length else ""

        if not in_single_quote and not in_double_quote and ch == "-" and nxt == "-":
            i += 2
            while i < length and sql_text[i] not in ("\n", "\r"):
                i += 1
            continue

        if ch == "'" and not in_double_quote:
            if in_single_quote and nxt == "'":
                buffer.append(ch)
                buffer.append(nxt)
                i += 2
                continue
            in_single_quote = not in_single_quote
            buffer.append(ch)
            i += 1
            continue

        if ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            buffer.append(ch)
            i += 1
            continue

        if ch == ";" and not in_single_quote and not in_double_quote:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer = []
            i += 1
            continue

        buffer.append(ch)
        i += 1

    tail = "".join(buffer).strip()
    if tail:
        statements.append(tail)

    return statements


def init_db(mysql_dsn: str) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    schema_sql = schema_path.read_text(encoding="utf-8")
    engine = make_engine(mysql_dsn)
    statements = _split_sql_statements(schema_sql)
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
