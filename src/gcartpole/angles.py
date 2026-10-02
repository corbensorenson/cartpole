"""Periodic angle coordinates that preserve representable errors near zero."""
import numpy as np


def wrap_angle(angle):
    values = np.asarray(angle, dtype=np.float64)
    wrapped = np.remainder(values + np.pi, 2.0 * np.pi) - np.pi
    # Adding pi can erase small angles or quantize them to pi's ULP.
    # An angle already in the canonical interval needs no arithmetic.
    return np.where((values >= -np.pi) & (values < np.pi), values, wrapped)
