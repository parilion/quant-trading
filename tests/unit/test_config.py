import pytest

from quant_trading.config import Settings


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pass@localhost:3306/quant")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    settings = Settings.from_env()

    assert settings.mysql_dsn == "mysql+pymysql://user:pass@localhost:3306/quant"
    assert settings.tushare_token == "test-token"
    assert settings.universe_index == "000905.SH"
    assert settings.top_k == 50
    assert settings.trade_cost_bps == 20


def test_settings_parse_numeric_values_from_env(monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pass@localhost:3306/quant")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setenv("TOP_K", "80")
    monkeypatch.setenv("TRADE_COST_BPS", "35")

    settings = Settings.from_env()

    assert settings.top_k == 80
    assert settings.trade_cost_bps == 35


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("TOP_K", "abc"),
        ("TRADE_COST_BPS", "x20"),
    ],
)
def test_settings_invalid_integer_value_has_field_name(monkeypatch, field, value):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pass@localhost:3306/quant")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setenv(field, value)

    with pytest.raises(ValueError, match=field):
        Settings.from_env()


def test_settings_top_k_must_be_positive(monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pass@localhost:3306/quant")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setenv("TOP_K", "0")

    with pytest.raises(ValueError, match="top_k"):
        Settings.from_env()


def test_settings_trade_cost_bps_must_be_non_negative(monkeypatch):
    monkeypatch.setenv("MYSQL_DSN", "mysql+pymysql://user:pass@localhost:3306/quant")
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")
    monkeypatch.setenv("TRADE_COST_BPS", "-1")

    with pytest.raises(ValueError, match="trade_cost_bps"):
        Settings.from_env()
