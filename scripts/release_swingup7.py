#!/usr/bin/env python
"""Regenerate the complete public seven-link release from one clean commit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gcartpole.evidence import file_sha256


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs/swingup7_uniform"
CONTROLLER = RUN_DIR / "seven_link_release_controller.json"
MANIFEST = RUN_DIR / "seven_link_swingup_manifest.json"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_clean_tracked_tree() -> str:
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        result = subprocess.run(["git", *args], cwd=ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError("tracked files must be clean before generating release evidence")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def rewrite_video_path(metadata_path: Path) -> None:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["video"]["path"] = "runs/swingup7_uniform/seven_link_swingup_success.mp4"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checksums() -> None:
    files = [
        RUN_DIR / "seven_link_swingup_manifest.json",
        RUN_DIR / "seven_link_release_controller.json",
        RUN_DIR / "eval_swingup7_20.json",
        RUN_DIR / "eval_swingup7_100.json",
        RUN_DIR / "seven_link_swingup_success.mp4",
        RUN_DIR / "seven_link_swingup_success.video.json",
        RUN_DIR / "robustness_sweep.json",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot checksum missing release files: {missing}")
    lines = []
    for path in files:
        name = Path(os.path.relpath(path, RUN_DIR)).as_posix()
        lines.append(f"{file_sha256(path)}  {name}")
    (RUN_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--twenty-seed", type=int, default=30732)
    parser.add_argument("--hundred-seed", type=int, default=40732)
    parser.add_argument("--video-seed", type=int, default=50732)
    args = parser.parse_args()
    source_commit = require_clean_tracked_tree()
    if not MANIFEST.is_file() or not CONTROLLER.is_file():
        raise FileNotFoundError("release manifest and controller must be committed before generation")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["evaluation"]
    actual = {
        "twenty_seed_start": args.twenty_seed,
        "hundred_seed_start": args.hundred_seed,
        "video_seed": args.video_seed,
    }
    for key, value in actual.items():
        if int(expected.get(key, -1)) != value:
            raise ValueError(f"manifest evaluation.{key} does not match requested release seed")

    common = [
        "--config", "configs/swingup7_uniform.yaml",
        "--spec", "benchmarks/p1_capture_envelope.yaml",
        "--controller", str(CONTROLLER.relative_to(ROOT)),
        "--manifest", str(MANIFEST.relative_to(ROOT)),
        "--prelude-seconds", "10",
        "--settle-mode", "hanging_lqr",
        "--settle-scale", "1.0",
        "--settle-control-cost", "1000",
        "--tracking-gain-scale", "2.0",
        "--shift-cart-nominal",
    ]
    with tempfile.TemporaryDirectory(prefix="swingup7-release-") as temporary:
        staging = Path(temporary)
        eval20 = staging / "eval_swingup7_20.json"
        eval100 = staging / "eval_swingup7_100.json"
        video = staging / "seven_link_swingup_success.mp4"
        video_meta = staging / "seven_link_swingup_success.video.json"
        for episodes, seed, out in (
            (20, args.twenty_seed, eval20),
            (100, args.hundred_seed, eval100),
        ):
            run([
                sys.executable,
                "scripts/evaluate_fddp_two_expert.py",
                *common,
                "--episodes", str(episodes),
                "--seed", str(seed),
                "--out", str(out),
            ])
        run([
            sys.executable,
            "scripts/render_fddp_two_expert_2d.py",
            "--config", "configs/swingup7_uniform.yaml",
            "--spec", "benchmarks/p1_capture_envelope.yaml",
            "--controller", str(CONTROLLER.relative_to(ROOT)),
            "--hanging-start",
            "--seconds", "30",
            "--seed", str(args.video_seed),
            "--prelude-seconds", "10",
            "--settle-mode", "hanging_lqr",
            "--settle-scale", "1.0",
            "--settle-control-cost", "1000",
            "--tracking-gain-scale", "2.0",
            "--shift-cart-nominal",
            "--fail-on-failure",
            "--out", str(video),
            "--metadata-out", str(video_meta),
        ])
        robustness = staging / "robustness_sweep.json"
        run([
            sys.executable,
            "scripts/evaluate_swingup7_robustness.py",
            "--episodes", "20",
            "--seed", "60732",
            "--out", str(robustness),
        ])
        rewrite_video_path(video_meta)
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        for path in (eval20, eval100, video, video_meta, robustness):
            shutil.copy2(path, RUN_DIR / path.name)
    write_checksums()
    print(f"Release regenerated from clean source commit {source_commit}")


if __name__ == "__main__":
    main()
