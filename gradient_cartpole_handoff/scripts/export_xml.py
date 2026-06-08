#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from gcartpole.config import apply_overrides, load_config
from gcartpole.morphology import build_morphology
from gcartpole.mjxml import generate_nlink_cartpole_xml


def main() -> None:
    parser = argparse.ArgumentParser(description="Export generated MuJoCo XML for a curriculum progress point")
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    morph = build_morphology(cfg["env"], cfg["morphology"], progress=args.progress)
    xml = generate_nlink_cartpole_xml(
        morph,
        cart_mass=float(cfg["env"]["cart_mass"]),
        rail_limit=float(cfg["env"]["rail_limit"]),
        force_limit=float(cfg["env"]["force_limit"]),
        timestep=float(cfg["env"]["timestep"]),
        cart_damping=float(cfg["env"].get("cart_damping", 0.0)),
        joint_armature=float(cfg["env"].get("joint_armature", 0.0)),
        link_radius=float(cfg["env"].get("link_radius", 0.025)),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    print(f"Wrote {out}")
    print("lengths:", morph.lengths)
    print("masses: ", morph.masses)
    print("damping:", morph.damping)


if __name__ == "__main__":
    main()
