from quant_trading.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pass@localhost:3306/quant")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    settings = Settings.from_env()

    assert settings.mysql_dsn == "mysql+pymysql://user:pass@localhost:3306/quant"
    assert settings.tushare_token == "test-token"
    assert settings.universe_index == "000905.SH"
