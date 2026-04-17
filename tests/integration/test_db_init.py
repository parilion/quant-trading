from pathlib import Path

import quant_trading.db.init_db as init_db_module
from quant_trading.db.schema import REQUIRED_TABLES


def test_required_tables_match_expected_plan_tables():
    expected_tables = {
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
    }
    assert set(REQUIRED_TABLES) == expected_tables
    assert len(REQUIRED_TABLES) == 10


def test_init_db_reads_schema_sql_and_executes_each_statement(monkeypatch):
    observed = {"read_path": None, "dsn": None, "executed": []}
    schema_sql = """
    CREATE TABLE foo (id INT);
    CREATE TABLE bar (id INT);
    """

    def fake_read_text(self, encoding="utf-8"):
        observed["read_path"] = self
        return schema_sql

    class FakeConn:
        def execute(self, statement):
            observed["executed"].append(str(statement))

    class FakeBeginContext:
        def __init__(self, conn):
            self._conn = conn

        def __enter__(self):
            return self._conn

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeEngine:
        def __init__(self):
            self._conn = FakeConn()

        def begin(self):
            return FakeBeginContext(self._conn)

    def fake_make_engine(mysql_dsn):
        observed["dsn"] = mysql_dsn
        return FakeEngine()

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(init_db_module, "make_engine", fake_make_engine)

    init_db_module.init_db("mysql+pymysql://user:pass@localhost:3306/quant")

    assert observed["dsn"] == "mysql+pymysql://user:pass@localhost:3306/quant"
    assert observed["read_path"] is not None
    assert observed["read_path"].name == "schema.sql"
    assert observed["executed"] == [
        "CREATE TABLE foo (id INT)",
        "CREATE TABLE bar (id INT)",
    ]


def test_split_sql_statements_handles_comments_and_semicolon_in_string():
    sql_text = """
    -- bootstrap comment
    CREATE TABLE foo (
      note VARCHAR(32) DEFAULT 'a;b'
    );

    ;   -- empty statement should be ignored
    -- next statement
    CREATE TABLE bar (id INT);
    """

    statements = init_db_module._split_sql_statements(sql_text)

    assert statements == [
        "CREATE TABLE foo (\n      note VARCHAR(32) DEFAULT 'a;b'\n    )",
        "CREATE TABLE bar (id INT)",
    ]
