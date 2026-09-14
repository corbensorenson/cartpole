from __future__ import annotations

import numpy as np
import pytest

from scripts.package_generalized_route import feedback_rms


def test_feedback_rms_is_shape_independent() -> None:
    gains = np.asarray([[3.0, 4.0], [0.0, 0.0]])
    assert feedback_rms(gains) == pytest.approx(2.5)


def test_feedback_rms_rejects_invalid_matrices() -> None:
    with pytest.raises(ValueError):
        feedback_rms(np.asarray([]))
    with pytest.raises(ValueError):
        feedback_rms(np.asarray([[np.nan]]))
