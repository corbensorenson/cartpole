import numpy as np
import pytest
from scipy import sparse
from spatialpole import Config,SpatialCartpole
from spatialpole.control import linearize_many
from spatialpole.direct_shooting import native_successors,shooting_defects,shooting_jacobian,apply_shooting_step,solve_augmented_ridge


@pytest.mark.parametrize('shape',[(8,5),(5,8)])
def test_augmented_ridge_matches_independent_dense_residual_solve(shape):
    rng = np.random.default_rng(786)
    m,n = shape
    u,_ = np.linalg.qr(rng.normal(size=(m,m)))
    v,_ = np.linalg.qr(rng.normal(size=(n,n)))
    s = np.zeros((m,n))
    np.fill_diagonal(s,np.geomspace(1e3,1e-6,min(m,n)))
    J,r,regularization = u@s@v.T,rng.normal(size=m),.01
    expected = np.linalg.lstsq(np.vstack((J,regularization*np.eye(n))),
                               np.r_[-r,np.zeros(n)],rcond=None)[0]
    actual = solve_augmented_ridge(sparse.csr_matrix(J),r,regularization)
    np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)


@pytest.mark.parametrize('n',[1,3])
def test_independent_shooting_intervals_match_serial_native_transitions(n):
    e = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(784)
    states = np.array([e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*.2) for _ in range(7)])
    controls = rng.normal(size=(7,2))*7.
    predicted = native_successors(e,states,controls)
    assert e.native_steps==len(controls)*e.config.frame_skip
    for state,force,result in zip(states,controls,predicted):
        np.testing.assert_array_equal(e.transition(state,force,quantize=False),result)
    e.close()


@pytest.mark.parametrize('fix_terminal',[False,True])
def test_sparse_manifold_defect_jacobian_matches_whole_native_residual(fix_terminal):
    e = SpatialCartpole(Config(n_links=2))
    rng = np.random.default_rng(785)
    H,nx = 5,2*e.model.nv
    states = np.array([e.integrate(e.upright_state(),rng.normal(size=nx)*.2) for _ in range(H+1)])
    controls = rng.normal(size=(H,2))*3.
    scales = np.r_[[3.,3.],np.full(e.model.nv-2,.15),[4.,4.],np.full(e.model.nv-2,15.)]
    residual,predicted = shooting_defects(e,states,controls,scales)
    models = linearize_many(e,states[:-1],controls)
    jac = shooting_jacobian(e,states,predicted,models,scales,fix_terminal)
    direction = rng.normal(size=jac.shape[1])*.1
    h = 1e-6
    hi,hu = apply_shooting_step(e,states,controls,direction,scales,h,fix_terminal)
    lo,lu = apply_shooting_step(e,states,controls,direction,scales,-h,fix_terminal)
    actual = (shooting_defects(e,hi,hu,scales)[0]-shooting_defects(e,lo,lu,scales)[0])/(2*h)
    np.testing.assert_allclose(jac@direction,actual.ravel(),atol=2e-7,rtol=1e-6)
    np.testing.assert_array_equal(hi[0],states[0])
    if fix_terminal:
        np.testing.assert_array_equal(hi[-1],states[-1])
    e.close()
