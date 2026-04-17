from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

STAGES = [
    "universe_snapshot",
    "raw_ingest",
    "clean_align",
    "label_build",
    "dataset_split",
    "fit_predict",
    "backtest",
    "evaluate_report",
]


def _run_stage(_stage_name: str, _run_id: str) -> dict[str, str]:
    return {"status": "ok"}


def run_pipeline(start_stage: str | None = None) -> dict[str, object]:
    if start_stage is None:
        start_index = 0
    else:
        if start_stage not in STAGES:
            raise ValueError(f"Unknown start_stage: {start_stage}")
        start_index = STAGES.index(start_stage)

    run_id = f"{datetime.now(tz=timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    stage_results: dict[str, dict[str, str]] = {}
    for stage_name in STAGES[start_index:]:
        stage_results[stage_name] = _run_stage(stage_name, run_id)

    return {
        "run_id": run_id,
        "start_stage": STAGES[start_index],
        "stages": stage_results,
    }
