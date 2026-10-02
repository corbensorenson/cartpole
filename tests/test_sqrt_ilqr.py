from types import SimpleNamespace
from dataclasses import replace

import mpmath as mp
import numpy as np

from gcartpole.ilqr import QuadraticTrajectoryCost, rollout
from gcartpole.sqrt_ilqr import optimize_sqrt_ilqr, square_root_backward_pass, square_root_tracking_gains


class LinearPlant:
    env = SimpleNamespace(n=1)

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self, x, u):
        return self.a @ x + self.b[:, 0] * u

    def linearize(self, x, u, **kwargs):
        return self.a, self.b

    def difference(self, x, reference):
        return x - reference


def test_mp_tracking_matches_independent_condensed_horizon_optimum():
    a = np.array([[1.2,.3],[0.,.9]])
    b = np.array([[.1],[.8]])
    plant = LinearPlant(a,b)
    cost = QuadraticTrajectoryCost(np.diag([2.,3.]),np.diag([5.,6.]),.2,100.,200.,0.,wrap_angles=False)
    horizon = 6
    factors = {}
    gains = square_root_tracking_gains(plant,np.zeros((horizon+1,2)),np.zeros(horizon),cost,
                                      regularization=.07,decimal_digits=80,
                                      value_factor_callback=lambda step,root:factors.__setitem__(step,root))
    for start in [0,2,5]:
        remaining = horizon-start
        x0 = np.array([.2,-.15])
        mapping = np.zeros((2,remaining));state=x0.copy();blocks=[];offsets=[]
        for step in range(remaining+1):
            root=np.linalg.cholesky(cost.terminal_state if step==remaining else cost.stage_state).T
            blocks.append(root@mapping);offsets.append(root@state)
            if step<remaining:
                mapping=a@mapping;mapping[:,step]+=b[:,0];state=a@state
        matrix=np.vstack(blocks+[np.sqrt(cost.control+.07)*np.eye(remaining)])
        rhs=np.r_[np.concatenate(offsets),np.zeros(remaining)]
        optimum=np.linalg.lstsq(matrix,-rhs,rcond=None)[0]
        np.testing.assert_allclose(gains[start]@x0,optimum[0],rtol=2e-13,atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(factors[start]@x0)**2,
                                   np.linalg.norm(matrix@optimum+rhs)**2,rtol=2e-13,atol=1e-15)


def test_mp_tracking_preserves_weak_value_directions_from_cost_factors():
    root=np.array([[1e20,1e20],[1.,0.],[0.,1.]])
    plant=LinearPlant(np.eye(2),np.array([[1.],[-1.]]))
    cost=QuadraticTrajectoryCost(np.eye(2),root.T@root,.1,100.,200.,0.,wrap_angles=False,terminal_factor=root)
    factors={}
    gains=square_root_tracking_gains(plant,np.zeros((2,2)),np.zeros(1),cost,regularization=0.,decimal_digits=80,
        value_factor_callback=lambda step,root:factors.__setitem__(step,root))
    # The strong row is orthogonal to the actuator. Its squared Gram entries
    # erase the unit directions in binary64, but those directions determine K.
    np.testing.assert_allclose(gains[0],[-1/2.1,1/2.1],rtol=2e-15)
    np.testing.assert_allclose(np.linalg.norm(factors[1]@np.array([1.,-1.]))**2,2.,rtol=2e-15)


def test_square_root_optimizer_matches_independent_dense_horizon_solution():
    a = np.array([[1.2, .3], [0., .9]])
    b = np.array([[.1], [.8]])
    plant = LinearPlant(a, b)
    cost = QuadraticTrajectoryCost(np.diag([2., 3.]), np.diag([5., 6.]),
                                   .2, 100., 200., 0., wrap_angles=False)
    x0 = np.array([.2, -.15])
    horizon = 6
    states, _ = rollout(plant, x0, np.zeros(horizon), cost, 1)
    blocks, offsets = [], []
    mapping = np.zeros((2, horizon))
    for step in range(horizon + 1):
        root = np.linalg.cholesky(cost.terminal_state if step == horizon else cost.stage_state).T
        blocks.append(root @ mapping)
        offsets.append(root @ states[step])
        if step < horizon:
            mapping = a @ mapping
            mapping[:, step] += b[:, 0]
    matrix = np.vstack(blocks + [np.sqrt(cost.control) * np.eye(horizon)])
    rhs = np.r_[np.concatenate(offsets), np.zeros(horizon)]
    expected = np.linalg.lstsq(matrix, -rhs, rcond=None)[0]
    solved = optimize_sqrt_ilqr(plant, x0, np.zeros(horizon), cost,
                               max_iterations=3, initial_regularization=0.)
    np.testing.assert_allclose(solved.controls, expected, atol=1e-8)
    assert solved.cost < rollout(plant, x0, np.zeros(horizon), cost, 1)[1]


def test_square_root_step_enforces_scalar_bound():
    plant = LinearPlant(np.eye(2), np.array([[1.], [0.]]))
    cost = QuadraticTrajectoryCost(np.eye(2), np.eye(2), .1, 100., 200., 0., wrap_angles=False)
    states = np.array([[10., 0.], [10., 0.]])
    delta, gains, active = square_root_backward_pass(plant, states, np.zeros(1), cost)
    np.testing.assert_array_equal(delta, [-1.])
    np.testing.assert_array_equal(gains, [[0., 0.]])
    assert active[0]


def test_ill_conditioned_residual_step_matches_high_precision_reference():
    root = np.array([[1e10, 1e10], [1., 0.], [0., 1.]])
    plant = LinearPlant(np.eye(2), np.ones((2, 1)))
    x = np.array([1., -1. + 1e-9])
    cost = QuadraticTrajectoryCost(np.eye(2), root.T @ root, .1,
                                   100., 200., 0., wrap_angles=False, terminal_factor=root)
    delta, gains, active = square_root_backward_pass(
        plant, np.array([x, x]), np.zeros(1), cost, regularization=0.,
    )
    with mp.workdps(80):
        exact_root = mp.matrix(root.tolist())
        exact_b = mp.matrix([1., 1.])
        rb = exact_root * exact_b
        rx = exact_root * mp.matrix(x.tolist())
        expected = -float((rb.T * rx)[0] / (mp.mpf(.1) + (rb.T * rb)[0]))
    np.testing.assert_allclose(delta[0], expected, atol=2e-16)
    assert np.all(np.isfinite(gains)) and not active[0]


def test_scheduled_factored_running_objective_matches_independent_dense_solution():
    plant=LinearPlant(np.array([[1.1,.2],[0.,.9]]),np.array([[.2],[.6]]))
    base=QuadraticTrajectoryCost(np.eye(2),np.diag([4.,5.]),.2,100.,200.,0.,wrap_angles=False,
                                 terminal_target=np.array([.03,0.]))
    root=np.array([[3.,-2.],[0.,2.]])
    capture=replace(base,stage_factor=root,stage_state=root.T@root,control=.7,
                    stage_target=np.array([.03,-.02]))
    horizon=6;costs=[base]*2+[capture]*4;x0=np.array([.15,.1]);mapping=np.zeros((2,horizon));x=x0.copy()
    blocks=[];rhs=[]
    for step in range(horizon+1):
        cost=base if step==horizon else costs[step]
        factor=np.linalg.cholesky(cost.terminal_state).T if step==horizon else (
            cost.stage_factor if cost.stage_factor is not None else np.linalg.cholesky(cost.stage_state).T)
        target=cost.terminal_target if step==horizon else cost.stage_target
        blocks.append(factor@mapping);rhs.append(factor@(x if target is None else x-target))
        if step<horizon:
            x=plant.a@x;mapping=plant.a@mapping;mapping[:,step]+=plant.b[:,0]
    controls_root=np.diag(np.sqrt([c.control for c in costs]))
    expected=np.linalg.lstsq(np.vstack(blocks+[controls_root]),-np.r_[np.concatenate(rhs),np.zeros(horizon)],rcond=None)[0]
    solved=optimize_sqrt_ilqr(plant,x0,np.zeros(horizon),base,running_costs=costs,
                              initial_regularization=0.,max_iterations=3)
    np.testing.assert_allclose(solved.controls,expected,atol=1e-8)
    np.testing.assert_allclose(solved.cost,rollout(plant,x0,solved.controls,base,1,running_costs=costs)[1],rtol=1e-14)
