from __future__ import annotations

import argparse

from quant_trading.pipeline.orchestrator import STAGES, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run quant trading pipeline")
    parser.add_argument(
        "--start-stage",
        dest="start_stage",
        choices=STAGES,
        default=None,
        help="Start pipeline from this stage",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_pipeline(start_stage=args.start_stage)

    print(f"run_id={result['run_id']}")
    for stage_name, payload in result["stages"].items():
        print(f"{stage_name}: {payload['status']}")
