from __future__ import annotations

import numpy as np
import pytest

from scripts.distill_swing_route_policy import parse_hidden_sizes, split_route_rows


def test_split_route_rows_is_nonempty_and_interleaved() -> None:
    train, validation = split_route_rows(15)
    assert np.array_equal(validation, np.array([0, 5, 10]))
    assert len(train) == 12
    assert not np.intersect1d(train, validation).size


def test_parse_hidden_sizes_rejects_empty_or_nonpositive_values() -> None:
    assert parse_hidden_sizes("256, 128") == [256, 128]
    with pytest.raises(ValueError):
        parse_hidden_sizes("")
    with pytest.raises(ValueError):
        parse_hidden_sizes("256,0")
