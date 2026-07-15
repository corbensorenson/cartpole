#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gcartpole.config import load_config
from gcartpole.evidence import git_metadata, runtime_metadata, utc_timestamp
from gcartpole.roadmap import (
    benchmark_snapshot,
    validate_canonical_config,
    validate_runtime_benchmark,
    validate_solution_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the ROADMAP.md benchmark and completion evidence")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--run-dir", default="runs/swingup7_uniform")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--static-only", action="store_true", help="Skip MuJoCo runtime assertions")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    cfg = load_config(config_path)
    errors = validate_canonical_config(cfg)
    snapshot = benchmark_snapshot(cfg) if not errors else {}
    if not errors and not args.static_only:
        runtime_errors, snapshot = validate_runtime_benchmark(cfg)
        errors.extend(runtime_errors)
    if not errors and not args.benchmark_only:
        errors.extend(validate_solution_artifacts(cfg, Path(args.run_dir), repo_root))

    report = {
        "generated_at": utc_timestamp(),
        "passed": not errors,
        "mode": "benchmark" if args.benchmark_only else "final_solution",
        "config": str(config_path),
        "snapshot": snapshot,
        "errors": errors,
        "runtime": runtime_metadata(),
        "git": git_metadata(repo_root),
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
