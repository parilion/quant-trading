import io
from contextlib import redirect_stdout

from quant_trading import cli
from quant_trading.pipeline.orchestrator import STAGES, run_pipeline


def test_stages_match_expected_order():
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


def test_run_pipeline_returns_run_id_and_stage_results():
    result = run_pipeline()

    assert isinstance(result["run_id"], str)
    assert result["run_id"]
    assert list(result["stages"].keys()) == STAGES

    for stage_name in STAGES:
        assert result["stages"][stage_name]["status"] == "ok"


def test_cli_accepts_start_stage_and_passes_to_orchestrator(monkeypatch):
    captured = {"start_stage": None, "execute": None}

    def fake_run_pipeline(start_stage=None, execute=False):
        captured["start_stage"] = start_stage
        captured["execute"] = execute
        return {"run_id": "test-run", "stages": {}}

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli.main(["--start-stage", "fit_predict"])

    assert captured["start_stage"] == "fit_predict"
    assert captured["execute"] is False


def test_cli_execute_flag_passed_to_orchestrator(monkeypatch):
    captured = {"execute": None}

    def fake_run_pipeline(start_stage=None, execute=False):
        captured["execute"] = execute
        return {"run_id": "test-run", "stages": {}}

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        cli.main(["--execute"])

    assert captured["execute"] is True
