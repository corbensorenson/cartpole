import numpy as np
import pytest
from spatialpole.transfer import regrid_planar_material_states


@pytest.mark.parametrize('source_count,target_count',[(14,13),(3,2),(1,3)])
def test_material_transfer_preserves_cart_and_rigid_unwrapped_motion(source_count,target_count):
    p = np.zeros((3,2*(source_count+1)))
    p[:,0],p[:,source_count+1] = [.1,.2,.3],[.4,.5,.6]
    p[:,1],p[:,source_count+2] = [-np.pi,-2*np.pi,-3*np.pi],[.2,.3,.4]
    original = p.copy()
    out = regrid_planar_material_states(p,np.ones(source_count),np.ones(target_count))
    np.testing.assert_array_equal(p,original)
    np.testing.assert_array_equal(out[:,0],p[:,0])
    np.testing.assert_array_equal(out[:,target_count+1],p[:,source_count+1])
    np.testing.assert_allclose(np.cumsum(out[:,1:target_count+1],axis=1),np.repeat(p[:,1:2],target_count,axis=1),atol=2e-15)
    np.testing.assert_allclose(np.cumsum(out[:,target_count+2:],axis=1),np.repeat(p[:,source_count+2:source_count+3],target_count,axis=1),atol=2e-15)


def test_material_transfer_matches_analytic_shape_and_rate_field():
    ns,nt = 14,13
    centers = (np.arange(ns)+.5)/ns
    target = (np.arange(nt)+.5)/nt
    angles,rates = -np.pi+.4*centers,.2+.3*centers
    physical = np.r_[.7,np.diff(np.r_[0.,angles]),.1,np.diff(np.r_[0.,rates])][None,:]
    out = regrid_planar_material_states(physical,np.ones(ns),np.ones(nt))[0]
    np.testing.assert_allclose(np.cumsum(out[1:nt+1]),-np.pi+.4*target,atol=2e-15)
    np.testing.assert_allclose(np.cumsum(out[nt+2:]),.2+.3*target,atol=2e-15)


def test_material_transfer_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        regrid_planar_material_states(np.zeros((2,6)),[1.,0.],[1.])
