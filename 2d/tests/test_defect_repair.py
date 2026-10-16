import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.control import linearize_many,difference_jacobian
from spatialpole.fddp import affine_backward,reference_gaps,gap_rollout,repair_reference,gap_transport
from spatialpole.transfer import split_reference_states
from spatialpole.physical import directions_and_rates


@pytest.mark.parametrize('track_actions',[False,True])
def test_affine_backward_matches_independent_condensed_optimum(track_actions):
    rng = np.random.default_rng(8241)
    H,nx,nu = 8,3,2
    A = np.array([[1.,0.1,0.],[0.,0.95,0.1],[0.,0.,0.8]])
    B = np.array([[0.01,0.],[0.1,0.01],[0.,0.15]])
    Q,R,P = np.diag([1.,2.,0.3]),0.2*np.eye(nu),5*np.eye(nx)
    initial = rng.normal(size=nx)
    xs,us = rng.normal(size=(H+1,nx)),rng.normal(size=(H,nu))
    reference = rng.normal(size=(H,nu)) if track_actions else np.zeros((H,nu))
    xs[0] = initial
    offsets = np.array([A@xs[t]+B@us[t]-xs[t+1] for t in range(H)])
    k,K,_ = affine_backward([(A,B)]*H,offsets,us,xs[:-1]@Q,np.repeat(Q[None],H,axis=0),P@xs[-1],P,R,0.,1000.,action_reference=reference if track_actions else None)
    x = initial.copy()
    controls = []
    for t in range(H):
        u = us[t]+k[t]-K[t]@(x-xs[t])
        controls.append(u)
        x = A@x+B@u
    # Independently eliminate dynamics and solve one full-horizon quadratic.
    base,mapping = initial.copy(),np.zeros((nx,H*nu))
    hessian,gradient = np.kron(np.eye(H),R),-(reference@R).ravel()
    for t in range(H+1):
        weight = P if t==H else Q
        hessian += mapping.T@weight@mapping
        gradient += mapping.T@weight@base
        if t<H:
            base,mapping = A@base,A@mapping
            mapping[:,t*nu:(t+1)*nu] += B
    expected = -np.linalg.solve(hessian,gradient).reshape(H,nu)
    np.testing.assert_allclose(controls,expected,atol=1e-10)


def test_state_and_control_trust_are_centered_on_the_current_proposal():
    rng = np.random.default_rng(8242)
    H,nx,nu = 8,3,2
    A = np.array([[1.,0.1,0.],[0.,0.95,0.1],[0.,0.,0.8]])
    B = np.array([[0.01,0.],[0.1,0.01],[0.,0.15]])
    Q,R,P = np.diag([1.,2.,.3]),.2*np.eye(2),5*np.eye(3)
    S,T = np.diag([.3,2.,.7]),.15
    initial = rng.normal(size=nx)
    xs,us = rng.normal(size=(H+1,nx)),rng.normal(size=(H,nu))
    xs[0] = initial
    offsets = np.array([A@xs[t]+B@us[t]-xs[t+1] for t in range(H)])
    k,K,_ = affine_backward([(A,B)]*H,offsets,us,xs[:-1]@Q,np.repeat(Q[None],H,axis=0),P@xs[-1],P,R,0.,1000.,S,T)
    actual,x = [],initial.copy()
    for t in range(H):
        u = us[t]+k[t]-K[t]@(x-xs[t])
        actual.append(u)
        x = A@x+B@u
    base,mapping = initial.copy(),np.zeros((nx,H*nu))
    hessian,gradient = np.kron(np.eye(H),R+T*np.eye(nu)),-T*us.ravel()
    for t in range(H+1):
        weight = P if t==H else Q
        hessian += mapping.T@(weight+S)@mapping
        gradient += mapping.T@(weight@base+S@(base-xs[t]))
        if t<H:
            base,mapping = A@base,A@mapping
            mapping[:,t*nu:(t+1)*nu] += B
    expected = -np.linalg.solve(hessian,gradient).reshape(H,nu)
    np.testing.assert_allclose(actual,expected,atol=1e-10)


def test_gap_forward_recovers_reference_and_closes_native_dynamics():
    e = SpatialCartpole(Config(n_links=2))
    initial = e.upright_state()
    H,nx = 12,2*e.model.nv
    us = np.tile([2.,-3.],(H,1))
    xs = [initial]
    for u in us:
        xs.append(e.transition(xs[-1],u,quantize=False))
    reference = np.array(xs)
    perturb = np.zeros(nx)
    perturb[[0,2,e.model.nv+3]] = [0.002,0.01,0.04]
    for t in range(1,H+1):
        reference[t] = e.integrate(reference[t],perturb*np.sin(t))
    _,predicted = linearize_many(e,reference[:-1],us,return_next=True)
    gaps = reference_gaps(e,reference,predicted)
    feed,gains = np.zeros_like(us),np.zeros((H,2,nx))
    for alpha in (0.,0.5,1.):
        trial,actions,fx = gap_rollout(e,initial,reference,us,gaps,feed,gains,alpha)
        actual_gaps = reference_gaps(e,trial,fx)
        np.testing.assert_allclose(actual_gaps,(1-alpha)*gaps,atol=1e-10)
        if alpha==0:
            for x,y in zip(trial,reference):
                np.testing.assert_allclose(e.difference(x,y),0.,atol=1e-10)
        if alpha==1:
            np.testing.assert_allclose(trial,xs,atol=1e-12)
    e.close()


def test_difference_jacobian_at_nonzero_manifold_defect():
    e = SpatialCartpole(Config(n_links=2))
    rng = np.random.default_rng(862)
    ref = e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*0.4)
    x = e.integrate(ref,rng.normal(size=2*e.model.nv)*0.2)
    d = rng.normal(size=2*e.model.nv)
    actual = (e.difference(e.integrate(x,d*1e-6),ref)-e.difference(e.integrate(x,-d*1e-6),ref))/2e-6
    np.testing.assert_allclose(difference_jacobian(e,x,ref)@d,actual,atol=1e-8)


def test_gap_transport_matches_complete_native_gap_preserving_forward_map():
    e = SpatialCartpole(Config(n_links=3))
    rng = np.random.default_rng(8523)
    x = e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*0.6)
    u = np.array([7.,-12.])
    models,pred = linearize_many(e,x[None],u[None],return_next=True)
    A,B = models[0]
    predicted = pred[0]
    reference = e.integrate(predicted,rng.normal(size=2*e.model.nv)*0.2)
    gap = e.difference(reference,predicted)
    offset = e.difference(predicted,reference)
    tangent,du = rng.normal(size=2*e.model.nv),rng.normal(size=2)
    eps = 1e-6
    hi = e.integrate(e.transition(e.integrate(x,tangent*eps),u+du*eps,quantize=False),gap)
    lo = e.integrate(e.transition(e.integrate(x,-tangent*eps),u-du*eps,quantize=False),gap)
    actual = (e.difference(hi,reference)-e.difference(lo,reference))/(2*eps)
    np.testing.assert_allclose(gap_transport(e,offset)@(A@tangent+B@du),actual,atol=1e-7,rtol=1e-6)
    shifted = e.integrate(predicted,(1-eps)*gap)
    np.testing.assert_allclose(e.difference(shifted,reference)/eps,offset,atol=1e-8)
    e.close()


def test_repair_rebuild_is_gap_free_on_native_one_link():
    e = SpatialCartpole()
    initial,_ = e.reset(seed=94,options={'angle_noise':0.01})
    states = np.repeat(e.upright_state()[None],31,axis=0)
    states[0] = initial
    repaired,actions,gains,design,history,proposal = repair_reference(e,initial,states,np.zeros((30,2)),iterations=4)
    assert np.isfinite(repaired).all() and np.max(np.linalg.norm(actions,axis=1)) <= e.config.force_limit
    for t in range(len(actions)):
        actual = e.transition(repaired[t],actions[t],quantize=False)
        np.testing.assert_array_equal(actual,repaired[t+1])
    assert history and min(r['max_scaled_defect'] for r in history) < 1e-8
    e.close()


@pytest.mark.parametrize('penalty',[1000.,100000.])
def test_tracking_repair_closes_native_defects(penalty):
    from spatialpole.tracking_repair import repair_tracking
    e = SpatialCartpole()
    initial = e.upright_state()
    controls = np.c_[np.sin(np.arange(20))*.2,np.zeros(20)]
    target = [initial]
    for u in controls:
        target.append(e.transition(target[-1],u,quantize=False))
    target = np.array(target)
    delta = np.zeros(2*e.model.nv)
    delta[3] = .01
    target[10] = e.integrate(target[10],delta)
    xs,us,_,_,history,_ = repair_tracking(e,initial,target,controls,iterations=6,penalty=penalty)
    for t,u in enumerate(us):
        np.testing.assert_array_equal(e.transition(xs[t],u,quantize=False),xs[t+1])
    assert min(r['max_scaled_defect'] for r in history)<1e-8
    assert np.max(np.abs(xs[:,:2]))<.01
    e.close()


def test_geometric_split_preserves_world_directions_and_rates_without_locks():
    old,new = SpatialCartpole(Config(n_links=3)),SpatialCartpole(Config(n_links=4))
    rng = np.random.default_rng(566)
    states = np.array([old.integrate(old.upright_state(),rng.normal(size=2*old.model.nv)) for _ in range(5)])
    d,v = directions_and_rates(old,states)
    for index in range(3):
        lifted = split_reference_states(old,new,states,index)
        ld,lv = directions_and_rates(new,lifted)
        mapping = list(range(3))
        mapping.insert(index+1,index)
        np.testing.assert_allclose(ld,d[:,mapping],atol=1e-12)
        np.testing.assert_allclose(lv,v[:,mapping],atol=1e-12)
    assert new.model.neq==0 and not new.model.jnt_limited.any()
