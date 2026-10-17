import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.transfer import (decode_planar_coordinates,declared_coordinate_projection,
                                  coordinate_projection,lift_planar_state)


@pytest.mark.parametrize('convention',['physical_relative','scaled_cumulative'])
def test_declared_donor_state_and_feedback_maps_preserve_physical_coordinates(convention):
    n = 4
    transform = np.eye(2*(n+1))
    if convention=='scaled_cumulative':
        transform[0,0],transform[n+1,n+1] = 1/1.25,1/.5
        transform[1:n+1,1:n+1] = np.tril(np.ones((n,n)))/.15
        transform[n+2:,n+2:] = np.eye(n)/.75
    rng = np.random.default_rng(894)
    physical = rng.normal(size=(5,2*(n+1)))
    physical[0] = 0.
    physical[0,1] = -np.pi
    encoded = physical@transform.T
    decoded = decode_planar_coordinates(encoded,transform,n)
    np.testing.assert_allclose(decoded,physical,atol=2e-15)
    env = SpatialCartpole(Config(n_links=n))
    hanging = lift_planar_state(env,decoded[0,:n+1],decoded[0,n+1:])
    env.set_state(hanging)
    np.testing.assert_allclose(env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2],
                               np.tile([0.,0.,-1.],(n,1)),atol=2e-15)
    projection = declared_coordinate_projection(env,transform)
    if convention=='scaled_cumulative':
        np.testing.assert_array_equal(projection,coordinate_projection(env))
    delta = rng.normal(size=2*(n+1))*1e-6
    ref = lift_planar_state(env,physical[1,:n+1],physical[1,n+1:])
    actual = lift_planar_state(env,physical[1,:n+1]+delta[:n+1],physical[1,n+1:]+delta[n+1:])
    np.testing.assert_allclose(projection@env.difference(actual,ref),transform@delta,atol=2e-15)
    env.close()


@pytest.mark.parametrize('problem',['shape','nonfinite','singular'])
def test_invalid_source_coordinate_maps_are_rejected(problem):
    transform = np.eye(6)
    if problem=='shape':
        transform = np.eye(4)
    elif problem=='nonfinite':
        transform[0,0] = np.nan
    else:
        transform[0] = 0.
    with pytest.raises(ValueError):
        decode_planar_coordinates(np.zeros((1,6)),transform,2)
