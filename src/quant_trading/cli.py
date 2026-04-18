from __future__ import annotations

import argparse

from dotenv import load_dotenv

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
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute real pipeline stages (requires DB + Tushare config).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    result = run_pipeline(start_stage=args.start_stage, execute=args.execute)

    print(f"run_id={result['run_id']}")
    for stage_name, payload in result["stages"].items():
        print(f"{stage_name}: {payload['status']}")
