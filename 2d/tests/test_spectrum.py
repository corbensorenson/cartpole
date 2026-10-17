from fractions import Fraction
import numpy as np
import pytest
from spatialpole.spectrum import feedback_spectral_radius


def test_mp_feedback_spectrum_uses_exact_input_product_and_preserves_matrices():
    A = np.diag([1e16,.5])
    B = np.array([[1e16],[0.]])
    K = np.array([[1.-2e-16,0.]])
    saved = [x.copy() for x in (A,B,K)]
    exact = float(Fraction.from_float(A[0,0])-Fraction.from_float(B[0,0])*Fraction.from_float(K[0,0]))
    assert exact!=float((A-B@K)[0,0])
    assert feedback_spectral_radius(A,B,K)==exact
    for actual,original in zip((A,B,K),saved):
        np.testing.assert_array_equal(actual,original)


def test_mp_feedback_spectrum_keeps_coupled_complex_block():
    A = np.array([[.3,-.4],[.4,.3]])
    assert abs(feedback_spectral_radius(A,np.zeros((2,1)),np.zeros((1,2)))-.5)<1e-15


def test_mp_spectrum_rejects_invalid_precision():
    with pytest.raises(ValueError,match='precision'):
        feedback_spectral_radius(np.eye(1),np.zeros((1,1)),np.zeros((1,1)),digits=15)
