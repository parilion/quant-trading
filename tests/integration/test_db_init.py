from quant_trading.db.schema import REQUIRED_TABLES


def test_required_tables_include_core_tables():
    required = {"meta_universe", "ads_pred_scores", "meta_run_log"}
    assert required.issubset(set(REQUIRED_TABLES))
