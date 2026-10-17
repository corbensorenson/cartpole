import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.control import linearize_many


@pytest.mark.parametrize('n',[1,3])
def test_fourth_order_batch_predicts_independent_spatial_directional_derivative(n):
    e = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(2031)
    nx = 2*e.model.nv
    state = e.integrate(e.upright_state(),rng.normal(size=nx)*.2)
    force = np.array([7.,-11.])
    models,next_state = linearize_many(e,state[None],force[None],eps=1e-3,return_next=True,order=4)
    A,B = models[0]
    np.testing.assert_array_equal(next_state[0],e.transition(state,force,quantize=False))
    direction,du = rng.normal(size=nx),rng.normal(size=2)
    h = 1e-6
    hi = e.transition(e.integrate(state,h*direction),force+h*du,quantize=False)
    lo = e.transition(e.integrate(state,-h*direction),force-h*du,quantize=False)
    expected = (e.difference(hi,next_state[0])-e.difference(lo,next_state[0]))/(2*h)
    np.testing.assert_allclose(A@direction+B@du,expected,atol=2e-7,rtol=1e-7)
    e.close()


@pytest.mark.parametrize('sign',[-1,1])
def test_fourth_order_saturated_force_column_agrees_with_inward_richardson(sign):
    e = SpatialCartpole(Config(n_links=2))
    rng = np.random.default_rng(2032)
    state = e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*.2)
    force = np.array([sign*80.,0.])
    _,B = linearize_many(e,state[None],force[None],eps=1e-3,order=4)[0]
    base = e.transition(state,force,quantize=False)
    derivatives = []
    for h in (1e-3,5e-4):
        inward = force-np.array([sign*h,0.])
        derivatives.append(-sign*e.difference(e.transition(state,inward,quantize=False),base)/h)
    np.testing.assert_allclose(B[:,0],2*derivatives[1]-derivatives[0],atol=1e-8,rtol=1e-6)
    e.close()
