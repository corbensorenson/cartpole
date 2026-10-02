"""Precision-preserving upright mechanics for the repository's passive planar chain.

Diagnostic only: parsed MuJoCo scalar errors and binary64 runtime arithmetic
remain. No nonlinear basin or hanging-start reachability certificate is implied.
The linearized mechanics are M qdd + D qd + G q = e_cart * force_limit * u.
At upright, COM horizontal derivatives are pivot-to-COM vertical distances;
G is the Hessian of gravitational potential in relative joint coordinates.
"""
import mpmath
import mujoco
import numpy as np


def upright_rk4_matrices(env, *, decimal_digits=100):
    """Return MP A/B and mechanics diagnostics, without finite differences.

    Requires the generated cart/+y-hinge serial tree with no springs, friction,
    equality constraints, contacts or fluid forces. Parsed model masses, COMs,
    inertia tensors, armature, damping and timestep are promoted exactly.
    The RK4 polynomial is applied once per native step with held cart force,
    then composed for frame_skip. This does not change native simulation.
    """
    model = env.model
    n, d = env.n, env.n+1
    if decimal_digits < 30:
        raise ValueError('at least thirty decimal digits required')
    if (model.nq != d or model.nv != d or model.nbody != n+2 or model.njnt != d
            or model.neq or model.na or model.nu != 1
            or model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4
            or model.opt.viscosity or model.opt.density
            or np.any(model.jnt_stiffness) or np.any(model.dof_frictionloss)
            or not np.array_equal(model.jnt_axis, np.vstack([[1.,0.,0.],np.tile([0.,1.,0.],(n,1))]))
            or np.any(model.jnt_pos) or not np.array_equal(model.qpos0,np.zeros(d))
            or not np.array_equal(model.body_parentid[1:],np.arange(n+1))
            or not np.array_equal(model.body_quat[1:],np.tile([1.,0.,0.,0.],(n+1,1)))
            or np.any(model.body_pos[1:,:2]) or np.any(model.body_ipos[1:,:2])
            or not np.array_equal(model.opt.gravity[:2],np.zeros(2))
            or model.opt.gravity[2] >= 0
            or model.actuator_trnid[0,0] != 0 or model.actuator_gear[0,0] != 1
            or np.any(model.actuator_gear[0,1:])):
        raise ValueError('requires the passive upright planar chain and RK4 held cart motor')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    if data.nefc:
        raise ValueError('upright equilibrium has active constraints')
    ctx = mpmath.mp.clone();ctx.dps=decimal_digits
    mp = ctx.mpf
    masses=[mp(float(v)) for v in model.body_mass[2:]]
    centers=[mp(float(v)) for v in model.body_ipos[2:,2]]
    lengths=[mp(float(v)) for v in model.body_pos[3:,2]]+[mp(0)]
    inertias=[]
    for quat,principal in zip(model.body_iquat[2:],model.body_inertia[2:]):
        w,x,y,z=[mp(float(v)) for v in quat]
        # y row of the inertial-to-body quaternion rotation.
        row=[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)]
        inertias.append(sum(row[k]**2*mp(float(principal[k])) for k in range(3)))
    jac = ctx.matrix(n,n)
    for i in range(n):
        for j in range(i+1):
            jac[i,j]=centers[i]+sum(lengths[j:i])
    mass=ctx.matrix(d,d);gravity=ctx.matrix(d,d)
    mass[0,0]=mp(float(model.body_mass[1]))+sum(masses)
    for j in range(n):
        mass[0,j+1]=mass[j+1,0]=sum(masses[i]*jac[i,j] for i in range(j,n))
        for k in range(n):
            mass[j+1,k+1]=sum(masses[i]*jac[i,j]*jac[i,k]+inertias[i]
                                  for i in range(max(j,k),n))
    for j in range(d):
        mass[j,j]+=mp(float(model.dof_armature[j]))
    weights=[-mp(float(model.opt.gravity[2]))*(masses[i]*centers[i]+lengths[i]*sum(masses[i+1:])) for i in range(n)]
    for j in range(n):
        for k in range(n):
            gravity[j+1,k+1]=-sum(weights[max(j,k):])
    damping=ctx.diag([mp(float(v)) for v in model.dof_damping])
    inv_mass=ctx.inverse(mass)
    continuous=ctx.matrix(2*d,2*d)
    for j in range(d):
        continuous[j,d+j]=1
    stiffness_block=-inv_mass*gravity;damping_block=-inv_mass*damping
    for j in range(d):
        for k in range(d):
            continuous[d+j,k]=stiffness_block[j,k]
            continuous[d+j,d+k]=damping_block[j,k]
    force=ctx.matrix(d,1);force[0]=mp(float(env.force_limit))
    input_acceleration=inv_mass*force
    augmented=ctx.matrix(2*d+1,2*d+1)
    for j in range(2*d):
        for k in range(2*d):
            augmented[j,k]=continuous[j,k]
    for j in range(d):
        augmented[d+j,2*d]=input_acceleration[j]
    z=augmented*mp(float(model.opt.timestep))
    step=ctx.eye(2*d+1)+z+z**2/2+z**3/6+z**4/24
    discrete=step**int(env.frame_skip)
    a=discrete[:2*d,:2*d];b=discrete[:2*d,2*d:]
    native_mass=np.zeros((d,d));mujoco.mj_fullM(model,native_mass,data.qM)
    mass64=np.array(mass.tolist(),dtype=float)
    difference=float(np.max(np.abs(mass64-native_mass)))
    if difference > 1e-12*max(1.,float(np.max(np.abs(native_mass)))):
        raise ValueError('derived mechanics disagree with native mass matrix')
    return a,b,dict(decimal_digits=decimal_digits,
        method='parsed-model COM mechanics, exact MP inversion and held-input RK4 polynomial',
        maximum_mass_difference_from_native=difference,
        mass_matrix=[[str(v) for v in row] for row in mass.tolist()],
        gravity_hessian=[[str(v) for v in row] for row in gravity.tolist()],
        input_precision='Parsed binary64 model scalars promoted exactly; no finite differences or rounded intermediate matrix algebra.',
        nonlinear_certified=False,simulation_precision_unchanged=True)
