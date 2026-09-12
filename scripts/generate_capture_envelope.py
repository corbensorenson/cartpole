#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from gcartpole.capture_envelope import generate_capture_states, validate_capture_states
from gcartpole.config import dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the frozen P1 synthetic capture envelope")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--out-dir", default="runs/p1_capture_envelope")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    out_dir = Path(args.out_dir)
    spec = load_config(spec_path)
    files = {}
    for split in spec["splits"]:
        payload = generate_capture_states(spec, split)
        errors = validate_capture_states(payload, spec, split)
        if errors:
            raise RuntimeError(f"generated invalid {split} split: {errors[:10]}")
        path = out_dir / f"{split}.json"
        dump_json(payload, path)
        files[split] = file_metadata(path)
        print(f"Wrote {split}: {path} ({payload['count']} states, sha256={files[split]['sha256']})")

    manifest = {
        "generated_at": utc_timestamp(),
        "benchmark": spec["name"],
        "generator": spec["generator"],
        "spec": {"path": str(spec_path), "sha256": data_sha256(spec)},
        "files": files,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(manifest, out_dir / "manifest.json")
    print(f"Wrote manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
