import numpy as np
import pytest
from spatialpole.symmetry import project_reflection_models
from spatialpole.affine_value import affine_tracking_values,reflection_tracking_values


def test_reflection_factored_values_match_full_qr_and_independent_condensed_optimum():
    rng = np.random.default_rng(937)
    H,nx = 4,6
    signs = np.array([1.,-1.,1.,-1.,1.,-1.])
    keep_A = signs[:,None]*signs[None,:]>0
    keep_B = signs[:,None]*np.array([1.,-1.])[None,:]>0
    models = [(np.where(keep_A,.5*np.eye(nx)+rng.normal(size=(nx,nx))*.03,0.)+rng.normal(size=(nx,nx))*1e-13,
               np.where(keep_B,rng.normal(size=(nx,2))*.1,0.)+rng.normal(size=(nx,2))*1e-13) for _ in range(H)]
    defects = rng.normal(size=(H,nx))*.02
    defects[:,signs<0] *= 1e-12
    saved = [(A.copy(),B.copy()) for A,B in models]
    previous_d = defects.copy()
    projected,d,record = project_reflection_models(models,defects,signs,np.ones(nx),force_limit=1.)
    assert record['max_normalized_A_change']<1e-10 and record['max_normalized_defect_change']<1e-10
    factors = [np.diag(np.arange(1.,nx+1))]*H
    terminal,R = np.diag(np.arange(2.,nx+2)),np.diag([2.,3.])
    feed,K,S,s = reflection_tracking_values(projected,d,factors,R,terminal,signs)
    full_feed,full_K,full_S,full_s = affine_tracking_values(projected,d,factors,R,terminal)
    np.testing.assert_allclose(feed,full_feed,atol=2e-14)
    np.testing.assert_allclose(K,full_K,atol=2e-14)
    for t in range(H+1):
        np.testing.assert_allclose(S[t].T@S[t],full_S[t].T@full_S[t],atol=2e-13)
        np.testing.assert_allclose(S[t].T@s[t],full_S[t].T@full_s[t],atol=2e-13)
    assert np.all(K[:,0,signs<0]==0) and np.all(K[:,1,signs>0]==0)
    assert np.all(feed[:,1]==0)
    x0 = rng.normal(size=nx)

    def residual(u):
        state,rows = x0.copy(),[]
        for t,action in enumerate(u.reshape(H,2)):
            rows.extend((factors[t]@state,np.sqrt(np.diag(R))*action))
            A,B = projected[t]
            state = A@state+B@action+d[t]
        rows.append(terminal@state)
        return np.concatenate(rows)

    zero = np.zeros(H*2)
    base = residual(zero)
    matrix = np.column_stack([residual(np.eye(H*2)[j])-base for j in range(H*2)])
    optimum = np.linalg.lstsq(matrix,-base,rcond=None)[0].reshape(H,2)
    state,actual = x0.copy(),[]
    for t in range(H):
        action = feed[t]-K[t]@state
        actual.append(action)
        A,B = projected[t]
        state = A@state+B@action+d[t]
    np.testing.assert_allclose(actual,optimum,atol=2e-13)
    assert np.max(np.abs(np.array(actual)[:,1]))>1e-3
    for (A,B),(old_A,old_B) in zip(models,saved):
        np.testing.assert_array_equal(A,old_A)
        np.testing.assert_array_equal(B,old_B)
    np.testing.assert_array_equal(defects,previous_d)


def test_reflection_model_guard_rejects_substantial_cross_dynamics():
    A = np.eye(2)
    A[0,1] = .001
    with pytest.raises(ValueError,match='exceeds'):
        project_reflection_models([(A,np.eye(2))],np.zeros((1,2)),np.array([1.,-1.]),np.ones(2))


def test_reflection_values_reject_coupled_action_cost():
    with pytest.raises(ValueError,match='diagonal'):
        reflection_tracking_values([(np.eye(2),np.eye(2))],np.zeros((1,2)),[np.eye(2)],
            np.array([[1.,.1],[.1,1.]]),np.eye(2),np.array([1.,-1.]))
