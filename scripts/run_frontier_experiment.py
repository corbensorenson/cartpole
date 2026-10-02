#!/usr/bin/env python
"""Run one bounded experiment with a frozen source/input snapshot and log."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("provide an experiment command after --")
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory / "snapshot"
    paths = []
    for folder, suffix in (("src", ".py"), ("scripts", ".py"),
                           ("tests", ".py"),
                           ("configs", ".yaml"), ("benchmarks", ".yaml")):
        paths.extend(sorted((ROOT / folder).rglob(f"*{suffix}")))
    paths.extend((ROOT / value).resolve() for value in args.input)
    paths.append(ROOT / "ROADMAP.md")
    paths.append(ROOT / "pyproject.toml")
    paths.append(ROOT / "requirements-mac.txt")
    paths.append(ROOT / "environment-aligator.yml")
    inputs = []
    for source in dict.fromkeys(paths):
        relative = source.relative_to(ROOT)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inputs.append(dict(path=str(relative), sha256=file_metadata(destination)["sha256"]))
    record = dict(schema_version=1, started_at=utc_timestamp(), status="running",
                  not_solution=True, command=command, cwd=str(ROOT),
                  environment={key: os.environ.get(key) for key in
                               ("PYTHONPATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
                  runtime=runtime_metadata(), git=git_metadata(ROOT), frozen_inputs=inputs)
    journal = directory / "execution.json"
    dump_json(record, journal)
    started = time.monotonic()
    with (directory / "output.log").open("w") as log:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,
                          status="finished" if result.returncode == 0 else "failed")
        except BaseException as error:
            record.update(status="interrupted_or_failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            record.update(finished_at=utc_timestamp(), wall_time_seconds=time.monotonic() - started)
            dump_json(record, journal)
    print(f"{record['status']}: {directory}", flush=True)
    raise SystemExit(record["returncode"])


if __name__ == "__main__":
    main()
