#!/usr/bin/env python
"""Measure missing-force acceleration on saved splines without rerunning search."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline

from gcartpole.config import dump_json
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.inverse_collocation import SplineInverseProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    source = json.loads(Path(args.controller).read_text())
    env = NLinkCartPoleEnv(source["effective_config"], progress=1., seed=0)
    data = source["search"]
    points = np.asarray(data["spline_control_points"])
    spline = BSpline(np.asarray(data["spline_knots"]), points, 5)
    duration = source["controller"]["horizon_seconds"]
    times = np.linspace(0., duration, 4 * (len(data["inverse_sample_times"]) - 1) + 1)
    problem = SplineInverseProblem(env.model, points[0], points[-1], duration,
                                  len(points), times, env.force_limit)
    errors = np.array([problem.acceleration_residual(q, v, a) for q, v, a in zip(
        spline(times), spline.derivative(1)(times), spline.derivative(2)(times))])
    dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
                   source=file_metadata(args.controller), runtime=runtime_metadata(),
                   max_missing_force_acceleration=float(np.max(np.abs(errors))),
                   rms_missing_force_acceleration=float(np.sqrt(np.mean(errors**2))),
                   max_missing_force_absolute_angular_acceleration=float(np.max(np.abs(np.cumsum(errors[:, 1:], axis=1)))),
                   dense_sample_count=len(times), inverse_calls=problem.inverse_calls,
                   note="Mixed cart m/s^2 and relative-joint rad/s^2 acceleration residual. Cart inverse force is unclipped here; source force limits and exact replay are reported separately."), args.out)
    env.close()


if __name__ == "__main__":
    main()
