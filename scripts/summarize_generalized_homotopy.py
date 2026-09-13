#!/usr/bin/env python
"""Publish a compact morphology/rail curve from a continuation ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, utc_timestamp


def curve_point(record: dict[str, Any]) -> dict[str, Any]:
    outcome = record.get("outcome", {})
    return {
        "trial": int(record["trial"]),
        "from_progress": float(record["from_progress"]),
        "proposed_progress": float(record["proposed_progress"]),
        "step": float(record["step"]),
        "accepted": bool(record["accepted"]),
        "termination_reason": outcome.get("termination_reason"),
        "max_upright_streak_seconds": outcome.get("max_upright_streak_seconds"),
        "max_cart_excursion": outcome.get("max_cart_excursion"),
        "rail_requirement": outcome.get("rail_requirement"),
        "length_fractions": record.get("dimensionless", {}).get("length_fractions"),
        "mass_fractions": record.get("dimensionless", {}).get("mass_fractions"),
    }


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    points = [curve_point(record) for record in payload.get("trials", [])]
    accepted = [point for point in points if point["accepted"]]
    rejected = [point for point in points if not point["accepted"]]
    accepted_rail = [
        float(point["rail_requirement"]["required_rail_ratio"])
        for point in accepted
        if point.get("rail_requirement") is not None
    ]
    rail_failures = [
        point for point in rejected if point["termination_reason"] == "rail_violation"
    ]
    return {
        "schema_version": 1,
        "claim_status": "development_measured_route_specific_rail_curve",
        "not_a_minimum_rail_certificate": True,
        "interpretation": (
            "Accepted ratios are body-aware rail requirements measured from exact "
            "replay. Rejected ratios describe failed-controller excursions and are "
            "not sufficient-rail estimates."
        ),
        "frontier": {
            "progress": float(payload["progress"]),
            "next_step": float(payload["next_step"]),
            "status": payload.get("status"),
            "trials": len(points),
            "accepted_trials": len(accepted),
            "rejected_trials": len(rejected),
            "rail_terminated_rejections": len(rail_failures),
            "latest_accepted_trial": accepted[-1]["trial"] if accepted else None,
            "latest_required_rail_ratio": accepted_rail[-1] if accepted_rail else None,
            "minimum_observed_accepted_rail_ratio": min(accepted_rail) if accepted_rail else None,
            "maximum_observed_accepted_rail_ratio": max(accepted_rail) if accepted_rail else None,
        },
        "accepted_curve": accepted,
        "rejected_curve": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    continuation_path = Path(args.continuation)
    payload = json.loads(continuation_path.read_text(encoding="utf-8"))
    output = summarize(payload)
    output["generated_at"] = utc_timestamp()
    output["continuation"] = file_metadata(continuation_path)
    dump_json(output, Path(args.out))
    print(
        f"trials={output['frontier']['trials']} "
        f"accepted={output['frontier']['accepted_trials']} "
        f"progress={output['frontier']['progress']:.9f}"
    )


if __name__ == "__main__":
    main()
