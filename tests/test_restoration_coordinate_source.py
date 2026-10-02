from pathlib import Path

import numpy as np
import pytest

from gcartpole.config import load_config
from gcartpole.evidence import file_metadata
from scripts.restore_frontier_trajectory import source_coordinate_transform


def test_legacy_transform_requires_matching_original_coordinate_spec():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root/"configs/swingup11_uniform.yaml")
    spec = root/"benchmarks/p1_capture_envelope.yaml"
    source = dict(controller={}, lyapunov=dict(coordinate_source=file_metadata(spec)))
    transform, provenance = source_coordinate_transform(source, cfg, spec)
    assert provenance == "verified_legacy_benchmark_spec"
    assert transform.shape == (24, 24)
    source["controller"]["coordinate_transform"] = transform.tolist()
    saved, provenance = source_coordinate_transform(source, cfg, spec)
    np.testing.assert_array_equal(saved, transform)
    assert provenance == "artifact"
    source["controller"].clear()
    source["lyapunov"]["coordinate_source"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="verified"):
        source_coordinate_transform(source, cfg, spec)
